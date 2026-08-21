# 072 - the global material stock and the block drop table

First ticket of iteration 2's economy (roadmap H2, and the half of "C3's
production/cost fields are read but inert" that can be made to mean
something without a simulation). This one builds the ledger and the
vocabulary; **nothing charges or credits it yet** — ticket 073 wires the
placement path, and production/warehouses come after that.

## The problem

The player has no materials. `Building::cost` parses and displays; digging a
hillside out from under a house destroys forty blocks of stone and nobody
records that they existed. There is no resource the game could deduct from
or add to, and no answer to "what *is* a resource here".

## The decisions

**Materials are Minecraft item ids** (the user's call). `Stock` is keyed by
`minecraft:oak_planks`, not by an invented `"wood"` — so a building's
existing `cost: [(block: "minecraft:oak_planks", count: 40)]` needs no schema
change, the world's own blocks are the economy's units, and production
chains later spell their goods the same way. The one thing this needs that
the vocabulary doesn't give for free is a **drop table**: `minecraft:stone`
cleared out of a hillside should stock cobblestone, the way mining it does,
and `minecraft:short_grass` should stock nothing at all.

**The drop table is a table, not a catalogue.** Every other data directory
here is one file per id (`assets/city/buildings/<id>.ron`). Drops are a
mapping over hundreds of block names, not a set of entities, so they get one
file: `assets/city/drops.ron`. Three rules, in order:

1. Air (`minecraft:air`/`cave_air`/`void_air`) drops nothing. Hardcoded, not
   a table entry — clearing air isn't a drop question and a table that
   forgets to say so would stock air by the thousand on every placement.
2. A name in `nothing` drops nothing.
3. A name in `replaced` drops what it says.
4. Anything else drops **itself, one for one** — the Minecraft default, and
   the reason a missing `drops.ron` is a working (if crude) economy rather
   than a dead one.

Properties are ignored: the key is `BlockState::name`, so an open door and a
closed one drop the same thing. Bare names in the file are namespaced the
same way `BlockState::from_str` does, so `stone` and `minecraft:stone` are
one key. A name in both lists is a load error — that's a contradiction, not
a precedence puzzle.

**Stock is integral (`u64`), and it is not the fractional accumulator.**
Production accrues at `per_minute` rates, but a rate that turns into 0.4 of
an oak plank in the stockpile makes every count a lie. Fractions belong in
the per-building buffer a later ticket adds; what reaches `Stock` is whole
units. Spending is all-or-nothing (`Stock::spend`) so a half-paid building
can't exist; removal is clamped at zero (`Stock::remove`), because a stock
that can go negative is a debt no mechanic here can discharge.

**Its own file, `<save>/citybuilder/stock.ron`.** Not a field on `city.ron`:
that file is at version 6 and its loader refuses anything else, so folding
the stock in would delete every existing city to add an empty ledger to it.
A separate file with its own version is the shape `journal.ron` already set,
and "no stock file yet" is honestly an empty stock rather than a quietly
defaulted field — the thing ticket 069's version note is actually against.

## Work

1. `city::inventory` — `Stock` (`Resource`, `BTreeMap<String, u64>`),
   `Parcel` (a bundle of items: what a spend consumed, what a clearing
   yielded, and later what a shipment carries), and `Stock::{count, add,
   add_parcel, remove, remove_parcel, can_afford, shortfall, spend, iter,
   is_empty}`. `spend` takes `&[definition::Cost]` and returns the `Parcel`
   it removed, so ticket 073's rollback path refunds exactly what it paid.
2. `city::drops` — `DropTable`, `DropTableError`, `load_drop_table`, and
   `DropTable::{drop_for, parcel_for}`. `parcel_for` takes an iterator of
   `&BlockState` so a `Baseline`'s `previous`/`written` can be run through it
   directly.
3. `city::inventory::{save_stock, load_stock, StockError}` —
   `<save>/citybuilder/stock.ron`, version 1, missing file is an empty
   stock, mirroring `journal::{save_journal, load_journal}` line for line.
4. `city::run` — load the table and the stock at startup (logged like every
   other load), insert both, and save the stock on `AppExit` in the same
   `Last` chain as `city.ron`/`journal.ron`.
5. `hot_reload::DefinitionErrors::drops` + a third section in the
   "Definition Errors" panel, seeded from the startup load. The file itself
   is **not** hot-reloaded — the snapshot machinery watches directories, and
   a one-file watcher is a ticket of its own if it turns out to be wanted.
6. `city::ui::city_panel` — a "Stock" section: every item and its count,
   sorted, short names (`oak_planks`, as the build menu already shows costs),
   `(nothing stockpiled)` when empty.
7. `assets/city/drops.ron` — the shipped table: fluids, plants, foliage,
   fire and snow drop nothing; stone/deepslate/grass-block/ores map to what
   mining them gives.

## Done when

- `cargo check` and `cargo test` pass.
- A save with no `stock.ron` opens with an empty stock, and a stock written
  on exit is read back identically next launch.
- The city panel shows the stock, and a broken `drops.ron` shows up in the
  errors panel instead of taking the game down.
