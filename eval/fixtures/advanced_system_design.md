---
fixture_id: advanced_system_design
material_type: document
expected_chunks: ~7
language: en
license: CC0 (self-authored for eval purposes)
---

# System Design Lecture: Building Services That Survive Scale

This is a transcript of a fictional graduate-level lecture on distributed system design. The content reads as a single spoken pass — the kind of material a Whisper-style audio extractor would produce from a recorded talk. There are occasional asides, mid-sentence corrections, and small redundancies, just as in real lectures.

---

All right, let's get started. Today we're going to walk through the building blocks that show up in just about every large-scale web service you've ever used: load balancers, caching layers, sharded storage, eventual consistency, the CAP theorem, and queue-based architectures. None of these ideas are new. What's interesting is how they fit together, and what each one costs you when you decide to use it.

I want to spend most of the time on trade-offs, because that's where students tend to lose the thread. Anyone can name "we use a cache" as an answer to a system design question. The harder skill is knowing when the cache is wrong, when it makes things worse, and what you signed up for the moment you put one in front of your database.

## Load balancers

Let's start with the front door. A load balancer's job is to take incoming traffic and spread it across a fleet of servers that all do roughly the same work. The simplest version is round-robin DNS — you publish multiple A records and let resolvers pick one. That works, sort of, but you have very little control: you can't drain a host quickly, you can't do health checks, and you depend on every client respecting your TTL, which they do not.

So in practice, you put a real load balancer in the path. Layer-4 balancers — think a TCP proxy — make routing decisions based on connection-level information: source and destination IP, port, maybe TLS SNI. They're fast, they don't need to terminate TLS, and they don't understand HTTP at all. Layer-7 balancers terminate the connection, look inside the request, and route on host, path, headers, cookies — anything you can parse. They're slower, they cost more, and they give you almost arbitrary routing policy in exchange.

The first lesson with load balancers is: health checks are the entire system. If your health check is `TCP connect succeeds`, you will route traffic to a process that has accepted connections but is internally on fire. You want a real `/healthz` that verifies the things your service depends on — the database connection, the cache, maybe a sample query — without being so heavyweight that it itself takes the host down. There's an art to this, and most teams get it wrong on the first three iterations.

The second lesson is: load balancing is also load shedding. When all backends are sick, the right answer is sometimes to start returning errors fast rather than to keep stuffing requests into queues that will time out anyway. Modern proxies — Envoy, HAProxy, NGINX in the right configuration — give you knobs for this. Use them.

## Caching tiers

Caching is the single most overused tool in distributed systems. It's also one of the most powerful. The reason it's overused is that it looks free. You put a Redis in front of your database, the slow queries get fast, and everyone is happy until the cache lies to you.

Let's talk about where caches live. You have, roughly, four tiers:

One. The browser cache. The request never leaves the user's machine. This is the cheapest cache there is. Set proper `Cache-Control` headers and you've already done more for latency than most clever server-side tricks.

Two. The CDN. A network of edge servers caches your responses geographically close to the user. CDNs are extremely good at static assets and increasingly good at dynamic content with short TTLs. The trade-off is invalidation: when something changes, you have to tell the CDN, and that's never instantaneous.

Three. The application-level cache. This is your Redis, your Memcached, your in-process LRU. It sits between your application servers and your database. It's the layer most people mean when they say "caching."

Four. The database's own cache — the buffer pool. This isn't usually something you reach for explicitly, but it's there, and a query that fits in the buffer pool is orders of magnitude cheaper than one that doesn't.

The hard problem with application caches is invalidation. The famous quote — "there are only two hard problems in computer science, cache invalidation and naming things" — is famous for a reason. You have a few patterns to choose from:

Cache-aside: the application checks the cache, falls back to the database on miss, and writes back. Simple, but you can have multiple processes racing to fill the same key, and you can have stale data lingering forever if writes don't invalidate.

Write-through: every write goes through the cache, which then writes to the database. The cache is never stale, but every write is slower, and the cache becomes a critical-path component.

Write-behind, also called write-back: writes hit the cache and queue up to flush to the database asynchronously. Fast, but you can lose data if the cache dies before the flush.

TTL-based: every entry expires after some time. Easy. The cost is that for that TTL window, you can serve stale data, and you have a thundering-herd problem when popular keys all expire at once.

In practice, most teams pick cache-aside with TTLs and add explicit invalidation on critical writes. It's a reasonable default. Just know what you signed up for.

## Sharding

Once a single database can no longer hold all your data — or more commonly, can no longer absorb all the writes — you shard. Sharding splits the data across multiple databases, each holding a slice. Reads and writes get routed to the right shard based on some key.

The sharding key is the most important decision you'll make. Get it right and the system grows linearly. Get it wrong and you have hot shards, cross-shard transactions, and a permanent migration problem. A few examples:

Hash-based sharding: `hash(user_id) mod N` picks the shard. Even distribution, almost no skew, but resharding when you change `N` rebalances most of the data. Consistent hashing fixes this — when a shard is added, only `1/N` of the keys move.

Range-based sharding: shard 1 holds users A-D, shard 2 holds E-H, and so on. Range queries are fast; "give me everyone whose name starts with B" doesn't have to scatter-gather. The downside is hot shards: if your data isn't uniformly distributed across the range, one shard does all the work.

Directory-based sharding: a lookup service maps each key to a shard. Maximum flexibility — you can move individual keys, you can rebalance based on load — but the lookup service is now critical infrastructure that you have to design just as carefully as the shards themselves.

There is no way to fully avoid the cross-shard problem. Some queries will need data from multiple shards, and now you have a distributed query engine, or you have your application stitching results together. Cross-shard transactions are even harder. Most teams either ban them — design the schema so transactions stay within one shard — or use a coordinating protocol like two-phase commit, which has its own failure modes.

## Eventual consistency

Strong consistency means every read sees the result of every write that finished before it. Eventual consistency means reads might see stale data, but if you stop writing, the system will converge.

The reason eventual consistency exists is that strong consistency across geography is expensive. To guarantee that every read in Tokyo sees a write that just happened in Frankfurt, that read has to wait for a round trip to Frankfurt, or for a globally agreed-upon order. There is no clever encoding that gets around this — it's a property of the network, plus the speed of light.

So we relax. We accept that for some kinds of data — a like count, a feed timestamp, a recommendation list — being a few hundred milliseconds behind is fine. In exchange, the user in Tokyo sees their request answered locally, fast.

The hard part is reasoning about what eventual consistency actually means for users. "Eventually" can be milliseconds, or it can be minutes. "Consistent" can mean every replica converges to the same value, or it can mean every read converges to monotonically newer values. Different storage systems offer different guarantees here, with names like read-your-writes, monotonic reads, causal consistency. They form a hierarchy, and it's worth knowing where the system you're using sits.

## CAP theorem

This is where I have to be careful, because the CAP theorem is one of the most misquoted ideas in our field. The actual claim is narrow: in the presence of a network partition, a distributed system has to choose between consistency and availability. It cannot guarantee both at the same time.

What the theorem does not say is "you must pick two of three at design time." Most production systems have no partition most of the time, and during normal operation they offer all three. The interesting question is what they do when a partition does happen. A CP system — say, a strongly consistent database — will refuse writes on the side that lost quorum, returning errors until the partition heals. An AP system — say, a Dynamo-style key-value store — will accept writes on both sides and reconcile later, accepting that some clients will see data that contradicts what the other side saw.

Neither choice is inherently better. They suit different problems. A bank ledger picks CP — it would rather refuse a transaction than double-spend. A shopping cart picks AP — losing a cart edit during a five-minute partition is annoying; refusing all cart writes during the same partition is unacceptable.

The corollary, which is sometimes called the PACELC extension, is that even when there's no partition, you are choosing between latency and consistency. A strongly consistent system has to wait for replicas; a relaxed-consistency system doesn't. So the "CAP question" really is: in the partition case, do you prefer C or A; and otherwise, do you prefer L or C. PACELC. It's a clunky acronym, but the underlying idea is sharp.

## Queue-based architectures

The last building block I want to cover is the message queue. Queues are how you decouple producers from consumers, smooth out spikes, and make work durable across crashes.

The mental model is simple: a producer puts a message on a queue. A consumer takes it off. The queue stores messages until they're consumed. Done.

The actual semantics get tricky almost immediately. Does the queue guarantee at-least-once delivery, at-most-once, or exactly-once? In practice, exactly-once over a network is impossible without help from the consumer — you can get effectively-exactly-once by making the consumer idempotent and giving it deduplication keys, but the queue alone cannot. Most modern queues — Kafka, SQS, RabbitMQ in the right mode — offer at-least-once and ask the consumer to deduplicate.

What does the queue do when the consumer is slow? Buffer up. What happens when the buffer fills? Some queues block producers; some drop messages; some fail over to disk. These behaviors are not interchangeable. If you choose a queue that drops messages under pressure and your producer is a payment processor, you have just designed a system that loses payments under load. This sounds obvious phrased that way; it is not obvious when you're picking between "AWS service A" and "AWS service B" on a comparison chart.

Queue-based architectures pair beautifully with eventual consistency. The producer writes a message and is done. The consumer applies the change to a downstream system, which becomes consistent shortly after. The two systems are never strongly consistent with each other, and that's fine because you designed for it.

The dark pattern here is using a queue to paper over a system that is too slow to keep up with traffic. If your queue depth grows unboundedly, you do not have a queue, you have a bug report. The queue is hiding the fact that you can't keep up. Monitor depth and oldest-message age. Alert when either grows beyond the SLA you're willing to commit to.

## Putting it together

A typical large-scale service looks like this. Requests hit a CDN, which serves what it can from cache. Misses go to a layer-7 load balancer, which routes to one of many stateless application servers. Those servers read from an application cache, falling back to a sharded primary database on miss, and write to that primary, which replicates asynchronously to read replicas in other regions. Long-running work is offloaded to a queue, processed by worker pools that update the database and any downstream caches. Critical operations carry idempotency keys end to end so that retries are safe.

Each layer in that picture is there for a reason, and each layer adds operational cost. A useful exercise — try this on your own designs — is to take each component out and ask: what would break, and could we live with that breaking? Often you'll find a layer you added because someone in a meeting said "we should cache this" without anyone asking what the failure mode of that cache would be when it goes down.

System design is mostly the discipline of taking things out of designs that don't need them, while leaving the parts that do. The building blocks we covered today are tools, not patterns to apply by default.

That's it for the lecture. Next time we'll do consensus protocols — Paxos, Raft, and why nobody implements either correctly on the first try.
