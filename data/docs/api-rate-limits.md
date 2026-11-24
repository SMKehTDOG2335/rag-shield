# API Rate Limits

The public REST API allows 100 requests per minute on the Free plan, 1,000 requests per minute on the Pro plan and 10,000 requests per minute on the Enterprise plan.

When a client exceeds its limit, the API responds with HTTP status 429 Too Many Requests and a Retry-After header indicating how many seconds to wait.

Rate limits are applied per API key, not per IP address. Batch endpoints count as a single request regardless of how many items they contain, up to 500 items per batch.

Clients should implement exponential backoff with jitter when retrying. Enterprise customers can request a temporary limit increase for planned migrations by contacting their account manager at least 5 business days in advance.
