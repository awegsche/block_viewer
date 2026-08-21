# 075 - interchangeable materials, so a birch forest can build an oak house

Follow-up to ticket 074, from the user's question: *"wood types are ignored,
right? if the player destroys a birch tree, they get only 'log' not
'birch_log'"*. They aren't — and that's the bug.

## The problem

`drops.ron`'s rule 4 (a block drops itself) keeps every wood type distinct,
which is right: the stock should say what you actually cut. But 074's
conversion table only maps each log to *its own* planks
(`birch_log -> birch_planks`), and the shipped house costs
`minecraft:oak_planks`. So a city in a birch forest stockpiles birch, reads
"short 40 more oak_planks", and can never build anything.

Two smaller gaps behind the same wall: `<type>_wood` (the six-sided bark
block), `stripped_<type>_log`/`stripped_<type>_wood`, and the bamboo family
all drop themselves and have no conversion at all, so they land in the pile
as dead weight.

## The decision

**Keep the types, make them fungible when a cost is paid.** The pile goes on
saying `birch_log`, because that's what the world gave; the *payment* stops
caring. A future building is still free to demand a specific wood, which the
alternative (collapsing everything to `oak_log` in the drop table) would have
made impossible to express.

**A group, not 132 conversion lines.** Twelve plank types made mutually
convertible is 132 ordered pairs; every wood-shaped block in the game is
worse. So `economy.ron` gains a second list next to `conversions`:

```ron
interchangeable: [
    ["oak_planks", "spruce_planks", "birch_planks", ...],
]
```

Every member of a group converts to every other **1:1**. Ratios stay in
`conversions` — a group is for materials that are the same material under a
different name, and a ratio means they aren't.

**Expanded at lookup, not at load.** Turning a 48-member group into 2,256
`Conversion` structs at load time would work and would be wasteful; instead
`plan_payment` asks "what routes lead to this item" and gets the explicit
conversions plus the asking item's group-mates. That keeps the group cheap
however large it grows, which matters because `city::ui::build_menu` prices
every visible row every frame.

Errors: a group with fewer than two members, a name listed twice inside one
group, and a name in two groups at once are all load errors — the last one
because "which group wins" is not a question the file should be able to ask.

## Work

1. `city::economy` — `EconomyConfig::groups`, `EconomyFile::interchangeable`,
   validation, and `Route`/`routes_to` so `cover` reads explicit conversions
   and group-mates through one list. `plan_payment` takes the whole
   `EconomyConfig` rather than a bare `&[Conversion]`.
2. `city::commit`, `city::ui::build_menu` — pass the config.
3. `assets/city/economy.ron` — two groups (every plank type; every log, bark
   block and stripped variant), plus the missing ratio conversions: bark and
   stripped logs to planks, bamboo, and a founding grant that stays as it is.

## Done when

- `cargo check` and `cargo test` pass.
- A stock holding only `birch_log` can pay a cost in `oak_planks`, and the
  build menu says what it will convert.
- A group with a name in it twice, or a name in two groups, is refused with
  a message in the errors panel rather than silently picking one.
