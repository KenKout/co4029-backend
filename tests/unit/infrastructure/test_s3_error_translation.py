"""How an S3 failure is classified, and which endpoint each caller is given.

``tests/integration/test_s3.py`` drives the happy paths against ``moto``.
What it cannot easily produce is the failure side: the ``ClientError``
branches that decide whether a missing object becomes a 404 or a 500, and
whether a permission failure is reported as such or buried in a generic
upload error. Those branches run inside ``except`` blocks, where a mistake
is doubly costly - it replaces the real S3 error with an error about
reading the error.

The endpoint helpers are here for the same reason: they are three lines
each and they decide whether a URL is reachable at all. A presigned URL
built on the internal endpoint points at a host the learner's browser
cannot resolve, and a worker sent to the public endpoint leaves the
cluster to fetch a file that was beside it.

Pure functions over dictionaries - no network, no ``moto``, no database.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError
from pydantic import SecretStr

from abridgeai.core.config import Settings
from abridgeai.infrastructure.errors import (
    S3NotConfiguredError,
    S3NotFoundError,
    S3PermissionError,
    S3UploadError,
)
from abridgeai.infrastructure.s3 import (
    _addressing_style,
    _browser_endpoint,
    _ensure_configured,
    _error_code,
    _http_status,
    _internal_endpoint,
    _s3_enabled,
    _translate_client_error,
)

_KEY = SecretStr("AKIAIOSFODNN7EXAMPLE")
_SECRET = SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "aws_access_key_id": _KEY,
        "aws_secret_access_key": _SECRET,
        "aws_endpoint_url": None,
        "aws_public_endpoint_url": None,
    }
    base.update(overrides)
    return Settings(**base)


def _client_error(code: str = "", status_code: int | str | None = None) -> ClientError:
    error: dict[str, Any] = {}
    if code:
        error["Code"] = code
    response: dict[str, Any] = {"Error": error}
    if status_code is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status_code}
    return ClientError(response, "GetObject")


class TestTheConfigurationGate:
    """Both halves of the credential pair are required, not either."""

    @pytest.mark.parametrize(
        ("key", "secret", "enabled"),
        [
            (_KEY, _SECRET, True),
            (_KEY, None, False),
            (None, _SECRET, False),
            (None, None, False),
        ],
    )
    def test_a_half_configured_deployment_counts_as_unconfigured(
        self, key: SecretStr | None, secret: SecretStr | None, enabled: bool
    ) -> None:
        """A key with no secret would fail later, at signing time, as a 403."""
        settings = _settings(aws_access_key_id=key, aws_secret_access_key=secret)
        assert _s3_enabled(settings) is enabled

    def test_the_refusal_names_both_variables_to_set(self) -> None:
        """This is read by whoever is deploying, so it has to be actionable."""
        settings = _settings(aws_access_key_id=None, aws_secret_access_key=None)
        with pytest.raises(S3NotConfiguredError) as caught:
            _ensure_configured(settings)
        message = str(caught.value)
        assert "AWS_ACCESS_KEY_ID" in message
        assert "AWS_SECRET_ACCESS_KEY" in message

    def test_a_configured_deployment_passes_silently(self) -> None:
        assert _ensure_configured(_settings()) is None


class TestWhichEndpointEachCallerGets:
    """The split exists because the two audiences sit on different networks.

    Presigned URLs are opened by the learner's browser, which is outside the
    cluster; server-side calls come from the backend and the workers, which
    are inside it. Handing either the other's endpoint produces a URL that
    resolves for nobody.
    """

    def test_the_browser_prefers_the_public_endpoint(self) -> None:
        settings = _settings(
            aws_endpoint_url="http://garage.internal:3900",
            aws_public_endpoint_url="https://files.example.edu",
        )
        assert _browser_endpoint(settings) == "https://files.example.edu"

    def test_the_browser_falls_back_to_the_only_endpoint_there_is(self) -> None:
        """Single-host dev: one endpoint serves both sides."""
        settings = _settings(aws_endpoint_url="http://localhost:3900")
        assert _browser_endpoint(settings) == "http://localhost:3900"

    def test_the_server_ignores_the_public_endpoint_entirely(self) -> None:
        """A worker must not leave the cluster to fetch a neighbouring file."""
        settings = _settings(
            aws_endpoint_url="http://garage.internal:3900",
            aws_public_endpoint_url="https://files.example.edu",
        )
        assert _internal_endpoint(settings) == "http://garage.internal:3900"

    def test_the_server_has_no_endpoint_when_only_a_public_one_is_set(self) -> None:
        """Falling back to the public URL here would route in-cluster traffic
        out through the internet and back."""
        settings = _settings(aws_public_endpoint_url="https://files.example.edu")
        assert _internal_endpoint(settings) is None

    @pytest.mark.parametrize("value", ["", None])
    def test_an_empty_endpoint_is_no_endpoint(self, value: str | None) -> None:
        """Empty string is what an unset environment variable arrives as, and
        it must resolve to ``None`` so boto reaches for the AWS default rather
        than trying to connect to ''."""
        settings = _settings(aws_endpoint_url=value, aws_public_endpoint_url=value)
        assert _browser_endpoint(settings) is None
        assert _internal_endpoint(settings) is None


class TestAddressingStyle:
    """Garage needs path-style; AWS wants virtual-host."""

    def test_a_custom_endpoint_forces_path_style(self) -> None:
        assert _addressing_style("http://garage.internal:3900") == "path"

    def test_no_endpoint_means_aws_and_virtual_host_style(self) -> None:
        """Path-style against AWS is deprecated and fails for newer buckets."""
        assert _addressing_style(None) == "virtual"


class TestReadingTheErrorItself:
    """These run inside ``except`` blocks and must never raise.

    An exception thrown while classifying a ``ClientError`` replaces the S3
    failure with an unrelated ``KeyError``, discarding both the cause and the
    action that produced it.
    """

    def test_a_well_formed_error_yields_code_and_status(self) -> None:
        exc = _client_error("NoSuchKey", 404)
        assert _error_code(exc) == "NoSuchKey"
        assert _http_status(exc) == 404

    def test_a_response_with_no_metadata_yields_zero_not_a_keyerror(self) -> None:
        exc = _client_error("NoSuchKey")
        assert _error_code(exc) == "NoSuchKey"
        assert _http_status(exc) == 0

    def test_a_response_with_no_error_block_yields_an_empty_code(self) -> None:
        exc = _client_error()
        assert _error_code(exc) == ""
        assert _http_status(exc) == 0

    def test_a_status_delivered_as_a_string_is_still_compared_as_a_number(self) -> None:
        """``404`` and ``"404"`` must classify the same way."""
        assert _http_status(_client_error("", "404")) == 404


class TestClassifyingTheFailure:
    """The three outcomes are not interchangeable.

    Not-found is an expected state the caller handles; permission-denied is a
    misconfiguration an operator must fix; anything else is a genuine upload
    failure. Collapsing them would make a broken bucket policy look like a
    transient error worth retrying forever.
    """

    @pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
    def test_every_spelling_of_missing_becomes_not_found(self, code: str) -> None:
        """S3 and Garage disagree about which of these they send."""
        with pytest.raises(S3NotFoundError):
            _translate_client_error(_client_error(code), action="head_object")

    @pytest.mark.parametrize("code", ["403", "AccessDenied", "Forbidden"])
    def test_every_spelling_of_denied_becomes_permission(self, code: str) -> None:
        with pytest.raises(S3PermissionError):
            _translate_client_error(_client_error(code), action="put_object_bytes")

    def test_an_unrecognised_code_is_classified_by_http_status(self) -> None:
        """A code the vocabulary does not know must not become a generic
        upload error when the status already says what happened."""
        with pytest.raises(S3NotFoundError):
            _translate_client_error(_client_error("SomethingNew", 404), action="head_object")
        with pytest.raises(S3PermissionError):
            _translate_client_error(_client_error("SomethingNew", 403), action="head_object")

    def test_anything_else_is_returned_rather_than_raised(self) -> None:
        """The two classified outcomes raise from inside; the general case is
        handed back for the caller to raise. A caller that assigned the result
        would never see a 404 or a 403 - which is why every call site spells
        it ``raise _translate_client_error(...)``.
        """
        returned = _translate_client_error(_client_error("InternalError", 500), action="upload")
        assert isinstance(returned, S3UploadError)
        assert not isinstance(returned, S3NotFoundError)
        assert not isinstance(returned, S3PermissionError)

    def test_the_action_is_named_in_every_message(self) -> None:
        """Every helper opens its own client, so the message is the only thing
        that says which call failed."""
        with pytest.raises(S3NotFoundError, match="download_to_temp"):
            _translate_client_error(_client_error("NoSuchKey"), action="download_to_temp")
        with pytest.raises(S3PermissionError, match="download_to_temp"):
            _translate_client_error(_client_error("AccessDenied"), action="download_to_temp")
        assert "download_to_temp" in str(
            _translate_client_error(_client_error("Throttled"), action="download_to_temp")
        )

    def test_a_featureless_error_still_produces_a_message(self) -> None:
        """Neither code nor status: the botocore text is better than nothing."""
        translated = _translate_client_error(_client_error(), action="upload")
        assert isinstance(translated, S3UploadError)
        assert str(translated) != "upload: "

    def test_the_original_error_is_kept_as_the_cause(self) -> None:
        """Without the chain, the botocore request id is lost from the log."""
        original = _client_error("AccessDenied", 403)
        with pytest.raises(S3PermissionError) as caught:
            _translate_client_error(original, action="head_object")
        assert caught.value.__cause__ is original
