---
fixture_id: beginner_python_loops
material_type: document
expected_chunks: ~5
language: en
license: CC0 (self-authored for eval purposes)
---

# Loops in Python: A Beginner Tutorial

Loops let a program repeat work without copying the same code over and over. Python ships with two loop statements, `for` and `while`, plus a small family of helper built-ins (`range`, `enumerate`) and control keywords (`break`, `continue`) that make the basics go a long way.

## The `for` loop

A `for` loop walks across a sequence one item at a time. Lists, tuples, strings, and ranges are all sequences in this sense.

```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

The variable `fruit` is rebound on every pass. After the loop finishes it still holds the last item, but you should not rely on that — treat the loop variable as scoped to the loop.

## `range` for counting

`range(stop)`, `range(start, stop)`, and `range(start, stop, step)` produce evenly-spaced integers. The end value is exclusive.

```python
for i in range(5):       # 0 1 2 3 4
    print(i)

for i in range(2, 10, 2):  # 2 4 6 8
    print(i)
```

`range` does not build a list in memory; it generates each integer on demand, so `range(10_000_000)` is cheap.

## `enumerate` for index + value

When you want both the position and the item, prefer `enumerate` over manual counters.

```python
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

`enumerate(fruits, start=1)` is handy when humans will read the output, since people number from one.

## The `while` loop

A `while` loop repeats as long as a condition is true. Use it when the number of iterations depends on something computed inside the loop.

```python
remaining = 10
while remaining > 0:
    print(f"counting down: {remaining}")
    remaining -= 1
```

If the condition never becomes false, the loop runs forever. That is the most common bug in `while` loops, so always make sure something inside the body changes the condition.

## `break` and `continue`

`break` exits the innermost loop immediately. `continue` skips the rest of the current pass and jumps to the next iteration.

```python
for n in range(20):
    if n == 7:
        break        # stop the whole loop
    if n % 2 == 0:
        continue     # skip even numbers
    print(n)         # prints 1 3 5
```

Use these sparingly. A loop body that is mostly `continue` chains is usually clearer rewritten as a guarded expression or a comprehension.

## When in doubt

Reach for `for` first. Use `while` only when you genuinely cannot pre-compute the sequence. Use `enumerate` instead of `range(len(...))`. And remember that idiomatic Python prefers comprehensions and the standard library's iterator tools over hand-rolled loops once the basics click.
