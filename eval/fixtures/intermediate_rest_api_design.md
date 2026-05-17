---
fixture_id: intermediate_rest_api_design
material_type: document
expected_chunks: ~5
language: en
license: CC0 (self-authored for eval purposes)
---

# REST API Design: Principles and Practices

Designing a REST API is less about choosing HTTP methods correctly and more about choosing the right level of abstraction. A good API hides the messy parts of your data model and exposes a small, stable surface that clients can rely on for years. This guide walks through the principles that hold up in production, the trade-offs each one forces, and a handful of common anti-patterns to avoid.

## Resources, Not Procedures

A REST API exposes resources — nouns — and lets clients act on them with a small set of verbs (HTTP methods). The temptation, especially when migrating from RPC-style services, is to expose endpoints like `POST /createUser`, `POST /getUserById`, and `POST /deleteUser`. That works, but you lose almost everything REST gives you: caching, intermediaries that understand idempotency, predictable URLs, and the ability to walk the API by following links.

Prefer:

```
GET    /users
POST   /users
GET    /users/{id}
PATCH  /users/{id}
DELETE /users/{id}
```

The `users` collection and individual `users/{id}` resources are the nouns. The HTTP methods are the verbs. Once a client understands one resource family, they can guess the shape of the next one — that consistency is the entire point.

## HTTP Methods and Their Contracts

Each method comes with a contract. Breaking the contract makes the API harder to use safely, especially at scale.

- **GET** retrieves a resource. It must be safe (no side effects) and idempotent. A `GET` should be cacheable by default.
- **POST** creates a new resource or triggers a non-idempotent action. The server picks the new resource's identity.
- **PUT** replaces a resource at a known URL. PUT is idempotent: sending the same payload twice leaves the server in the same state.
- **PATCH** applies a partial update. PATCH is not necessarily idempotent — that depends on the patch format. JSON Merge Patch is idempotent; JSON Patch with `add` operations is not.
- **DELETE** removes a resource. DELETE is idempotent: deleting an already-deleted resource should still return success (or 404, depending on your taste).
- **HEAD** is `GET` without a body — useful for cache validation and existence checks.
- **OPTIONS** describes what methods a resource supports; mostly used by CORS preflight.

```bash
# Create a user
curl -X POST https://api.example.com/v1/users \
  -H 'Content-Type: application/json' \
  -d '{"email":"ada@example.com","name":"Ada Lovelace"}'

# Replace a user
curl -X PUT https://api.example.com/v1/users/42 \
  -H 'Content-Type: application/json' \
  -d '{"email":"ada@example.com","name":"Augusta Ada King"}'

# Patch one field
curl -X PATCH https://api.example.com/v1/users/42 \
  -H 'Content-Type: application/merge-patch+json' \
  -d '{"name":"Countess of Lovelace"}'
```

### Don't: tunnel everything through POST

Tunneling reads and updates through `POST` (e.g. `POST /users/42/update`) breaks caching, breaks intermediaries that retry idempotent requests, and makes API exploration harder. Reach for the right verb even if your framework makes the wrong one slightly easier.

## Status Codes

Status codes are the first signal a client uses to decide what to do. Using them precisely costs nothing and saves clients from parsing error bodies just to figure out the request was rejected.

A short, opinionated list:

- **200 OK** — successful read or update with a body.
- **201 Created** — successful creation. Include a `Location` header pointing at the new resource.
- **202 Accepted** — the work has been queued; check back later. Pair with a job-status resource.
- **204 No Content** — successful, no body to return (typical for `DELETE` and some `PUT`s).
- **301/308** — permanent redirect, with 308 preserving the method and body.
- **400 Bad Request** — client sent malformed input.
- **401 Unauthorized** — client did not authenticate; include a `WWW-Authenticate` challenge.
- **403 Forbidden** — client authenticated but is not allowed.
- **404 Not Found** — the resource does not exist (or the client should not learn it does).
- **409 Conflict** — the request collided with another change (optimistic concurrency, duplicate creation).
- **422 Unprocessable Entity** — input parsed fine but failed semantic validation.
- **429 Too Many Requests** — rate limit hit; include `Retry-After`.
- **500 Internal Server Error** — server bug.
- **503 Service Unavailable** — temporary overload or maintenance; include `Retry-After`.

### Don't: return 200 with `{"error": ...}`

A 200 response with an error body forces every client to inspect the body just to know if the call succeeded. It also defeats every intermediate cache, retry layer, and monitor that expects HTTP status to mean what it says.

## Idempotency

Idempotency is the property that lets a client safely retry a request after a network blip without risking double-charging a credit card or creating two of the same record. `GET`, `PUT`, and `DELETE` are idempotent by spec; `POST` typically is not.

For non-idempotent operations, expose an idempotency key. Stripe popularized the pattern, and it has become the default:

```
POST /v1/payments
Idempotency-Key: 7f9b3e22-4f51-4a7e-8a5e-b6e17e4f0c11
```

The server stores `(key, request hash) -> response` for some retention window (24 hours is common). A retry with the same key returns the original response without re-running the side effect. A retry with the same key but a different body should return 422 — the client is confused, and the server should refuse to make a guess.

## Versioning

APIs change. Versioning is how you let them change without breaking the clients that already depend on them.

The two mainstream choices:

1. **URL versioning** — `/v1/users`, `/v2/users`. Easy to read, easy to route, easy for caches and CDNs to key on. The downside is that bumping the version forces every URL to change, which discourages making small breaking changes (so you batch them, which makes upgrades worse).
2. **Header versioning** — `Accept: application/vnd.example.v1+json`. Cleaner URLs, more flexibility, but harder to test from a browser and harder for caches that don't vary on `Accept`.

Pick one and stick with it. URL versioning is the more common choice for public APIs because it is the easiest for callers to learn.

Within a version, prefer additive changes: add a new field, a new endpoint, a new optional query parameter. Removing a field, renaming a field, or changing a field's type without a version bump is a breaking change, even if no client you know of uses it.

### Don't: ship `?version=2` as a query parameter

Query strings are widely cached at the wrong granularity, and clients tend to forget to include them on follow-up calls. Either version in the URL path or in a header.

## Pagination

Any collection that can grow without bound needs pagination. Two patterns dominate:

- **Offset/limit** — `GET /users?offset=200&limit=50`. Easy to implement, easy to understand, and broken once the data behind it changes during pagination (you skip or duplicate items).
- **Cursor** — `GET /users?cursor=eyJ...&limit=50`. The server returns an opaque cursor that encodes the position. New writes do not shift previously-paginated rows. This is the right default for anything backed by a mutable dataset.

Always include a `Link` header (or a `next`/`prev` field) with the URLs for the next page. Forcing clients to construct cursor URLs by hand is a recipe for off-by-one bugs in their code.

## Filtering, Sorting, Sparse Fieldsets

Once a collection has more than a screen of items, clients want to slice it. Stick to query parameters, and standardize on names:

- **Filtering** — `?status=active&role=admin`. For ranges and complex predicates use bracket syntax (`?created_at[gte]=2024-01-01`) or a dedicated filter language; either is fine, but be consistent.
- **Sorting** — `?sort=-created_at,name`. A leading `-` means descending. Allow exactly the fields you have indexes on; a sort that triggers a full table scan is a denial-of-service vector.
- **Sparse fieldsets** — `?fields=id,email,name`. Lets clients trim payloads they do not need. JSON:API standardized one shape for this; pick yours and document it.

## Error Bodies

A status code tells the client what category the failure belongs to. The body should tell them how to fix it. RFC 7807 (`application/problem+json`) is a good default:

```json
{
  "type": "https://api.example.com/errors/email-taken",
  "title": "Email already in use",
  "status": 409,
  "detail": "The email ada@example.com is registered to another account.",
  "instance": "/v1/users",
  "field_errors": [
    {"field": "email", "code": "duplicate"}
  ]
}
```

Stable machine-readable codes (here `email-taken`, `duplicate`) matter more than human strings; clients build retry logic against them.

## Authentication and Authorization

Use HTTPS everywhere — there is no acceptable reason to ship a production API on plain HTTP in 2024. Pick one auth scheme per audience:

- **OAuth 2.0 / OIDC** for third-party integrations.
- **API keys** for server-to-server traffic where the caller is your own infrastructure or a trusted partner.
- **Short-lived bearer tokens** (JWT or opaque) for first-party clients, refreshed via a dedicated endpoint.

Keep the auth concern out of every individual endpoint's documentation by handling it uniformly at the gateway. The endpoint docs should say "requires `users:read` scope" and stop there.

## HATEOAS, Lightly Applied

HATEOAS — Hypermedia as the Engine of Application State — is the part of REST that most APIs do not implement, and that is fine. The full vision (clients discover everything from links and never hardcode URLs) is rarely worth the engineering cost.

The light version, however, is almost always worth it: include links to related resources and to the next state transitions in your responses.

```json
{
  "id": 42,
  "status": "pending_review",
  "_links": {
    "self":     {"href": "/v1/orders/42"},
    "approve":  {"href": "/v1/orders/42/approve"},
    "cancel":   {"href": "/v1/orders/42/cancel"},
    "customer": {"href": "/v1/customers/7"}
  }
}
```

Clients still hardcode the entry points, but they can follow links from there. The big payoff is that the server can change URL shapes without breaking clients, and clients can learn what actions are currently allowed without parsing your state machine into their own code.

## Caching

`GET` responses should be cacheable by default. Set `Cache-Control` and `ETag` headers, and respect `If-None-Match` to send `304 Not Modified`. The hard part is choosing the right TTL: too short and you lose the benefit, too long and clients see stale data.

For private, user-specific data, set `Cache-Control: private` so shared caches do not serve one user's data to another. For collections that change often, lean on `ETag` rather than expiry — clients revalidate cheaply, and a 304 round-trip beats a full payload.

## Rate Limiting

Every public endpoint needs a rate limit. Beyond protecting the server, well-communicated limits help clients build their own backpressure:

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 873
X-RateLimit-Reset: 1701389400
```

When the client crosses the line, return 429 with `Retry-After`. Per-endpoint limits matter more than global ones — a slow `POST /search` should not exhaust the quota for cheap `GET`s.

## Designing for Change

The single best predictor of an API's longevity is whether the team behind it treats every breaking change as a serious cost. The principles in this document — resources over RPC, precise status codes, idempotency keys, additive evolution, hypermedia links — are all in service of that. Each one buys you the ability to change something on the server without forcing every client to redeploy.

You will not get every detail right on day one, and that is fine. Document what you have, version what you ship, and treat your API as a contract whose stability is itself a feature.
