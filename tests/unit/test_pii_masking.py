"""PII masking for admin aggregate views (PRD ADM-024).

The masks have a job beyond hiding characters: an operator scanning a list must
still be able to say "these rows are the same tenant" or "these forty failures
share a network". So the domain survives in an email and the /16 survives in an
IPv4 address, while the parts that identify a person or a subscriber do not.

These also pin the failure modes, which is where masking usually leaks: short
local parts, values that are not addresses at all, and None.
"""

import pytest

from abridgeai.core.security.pii import mask_email, mask_ip


class TestMaskEmail:
    def test_keeps_two_characters_and_the_domain(self) -> None:
        assert mask_email("nam.nguyen@example.com") == "na***@example.com"

    def test_domain_survives_so_tenants_stay_groupable(self) -> None:
        # The point of keeping it: "everyone here is from one org" is the
        # pattern an operator is scanning for.
        masked = mask_email("someone@acme.edu.vn")
        assert masked is not None
        assert masked.endswith("@acme.edu.vn")

    def test_two_character_local_part_is_not_passed_through(self) -> None:
        # A naive prefix slice would return the entire local part here.
        assert mask_email("ab@example.com") == "a***@example.com"

    def test_single_character_local_part(self) -> None:
        assert mask_email("a@example.com") == "a***@example.com"

    def test_value_that_is_not_an_address_is_fully_masked(self) -> None:
        # Whatever it is, it came out of a column holding personal data.
        assert mask_email("not-an-email") == "***"

    def test_none_passes_through_as_none(self) -> None:
        # Absent and redacted are different states and callers rely on that.
        assert mask_email(None) is None

    def test_empty_string_passes_through(self) -> None:
        assert mask_email("") == ""

    def test_never_leaks_the_full_local_part(self) -> None:
        for address in (
            "administrator@example.com",
            "j.smith@example.com",
            "x@example.com",
        ):
            masked = mask_email(address)
            local = address.split("@")[0]
            assert masked is not None
            if len(local) > 1:
                assert local not in masked


class TestMaskIp:
    def test_ipv4_keeps_the_first_two_octets(self) -> None:
        assert mask_ip("14.224.10.7") == "14.224.xxx.xxx"

    def test_ipv4_host_portion_is_gone(self) -> None:
        masked = mask_ip("203.0.113.42")
        assert masked is not None
        assert "113" not in masked
        assert "42" not in masked

    def test_two_addresses_on_one_network_mask_identically(self) -> None:
        # This is what makes "these share a network" readable in a list.
        assert mask_ip("14.224.10.7") == mask_ip("14.224.99.250")

    def test_different_networks_stay_distinguishable(self) -> None:
        assert mask_ip("14.224.10.7") != mask_ip("14.225.10.7")

    def test_ipv6_keeps_two_groups(self) -> None:
        masked = mask_ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert masked is not None
        assert masked.startswith("2001:0db8:")
        assert "7334" not in masked

    def test_unparseable_value_is_fully_masked(self) -> None:
        # Failing to parse is not evidence that it is safe to publish.
        assert mask_ip("not-an-ip") == "***"

    def test_none_passes_through_as_none(self) -> None:
        assert mask_ip(None) is None

    @pytest.mark.parametrize("address", ["10.0.0.1", "192.168.1.1", "8.8.8.8"])
    def test_never_returns_the_input_unchanged(self, address: str) -> None:
        assert mask_ip(address) != address
