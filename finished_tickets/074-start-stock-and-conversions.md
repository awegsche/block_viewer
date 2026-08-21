# 074 - a founding stock, and materials that convert on the way to a cost

Closes the bootstrap gap ticket 073 shipped with (a new city could earn only
what it dug, and nothing it could earn bought the shipped house), and adds
the user's second ask: a table of **trivially transformable** materials —
`1 oak_log -> 4 oak_planks` — applied automatically when a placement is paid
for.

## The problem

Two, with one config file between them:

1. **Nothing grants a starting stock.** A fresh save opens with an empty
   pile, so the first building is unaffordable and stays that way until the
   player terraforms enough ground to pay for it — and terraforming can't
   produce planks at all.
2. **Costs are spelled in refined materials, stock arrives raw.** A player
   holding sixty logs and no planks can't build a house that wants planks,
   even though the two are the same material one crafting step apart. Asking
   them to notice that and click a "convert" button is busywork; the game
   should just do it.

## The decisions

**One config file, `assets/city/economy.ron`, holding both.** Not two: they
are both small tables of numbers the player is expected to hand-tune ("I will
update it as the game progresses"), and one file means one loader, one error
line in the panel, and one place to look. `drops.ron` stays separate — it's a
mapping over hundreds of *block* names, a different kind of thing from a
dozen economy knobs.

**The grant applies when there is no `stock.ron`, not when the stock is
empty.** A player who spends everything has not founded a new city and must
not be refilled. That means [`inventory::load_stock`] has to distinguish
"file absent" from "file present and empty" — it now returns
`Option<Stock>`, and `city::run` seeds the grant on `None`. A stock file that
fails to *load* (corrupt, wrong version) is deliberately **not** granted
either: there was a stock there, and handing out a fresh one would turn a bad
file into free materials.

**Conversions cover a shortfall; they don't run on their own.** Nothing
converts the whole pile up front. When a placement is priced, each item the
stock is short of is covered by converting whatever the table says makes it,
in whole runs — needing 2 planks with a log in the pile converts the whole
log and leaves the other 2 planks in stock. Consequences:

- **Placement only.** Demolition's backfill and terraforming's fill *debit*,
  and debits clamp (073); converting materials to satisfy a clamp would be
  the game spending the player's logs to fill a hole. Costs are the one place
  a shortfall is worth solving rather than absorbing.
- **Chains work, cycles can't hang.** Covering `sticks` may need `planks`
  which may need `logs`, so the search recurses — with a depth cap and a
  visited set, so a table that says `A -> B` and `B -> A` refuses to loop
  rather than hanging the frame. `from.item == to.item` is a load error.
- **The conversion is part of the placement's ledger.** What it consumed
  lands in `debited`, what it produced in `credited`, alongside the cost and
  the terrain drops — so undo reverses the conversion too and the pile comes
  back as logs, not as planks. A rolled-back write (the apply failed) undoes
  it the same way.
- **The build menu prices with conversions.** Otherwise a row reads red while
  the click that follows it succeeds.

## Work

1. `city::economy` — `EconomyConfig` (`start_stock: Parcel`, `conversions:
   Vec<Conversion>`), `Conversion`/`ItemStack` (`count` defaults to 1),
   `load_economy`, `EconomyError`, and the planner:
   `plan_payment(stock, costs, conversions) -> Payment { conversion, shortfall }`
   — one function both the build menu and `city::commit` read, so the menu
   can never disagree with what the click does.
2. `inventory::load_stock` -> `Result<Option<Stock>, StockError>`;
   `Parcel::add_all`.
3. `city::run` — load the economy config, seed the grant when there's no
   stock file, log both.
4. `city::commit` — plan the payment, refuse with the post-conversion
   shortfall, apply the conversion at the same instant the cost is spent,
   carry `gained` on `PendingCommit` so a failed apply reverses it.
5. `city::ui::build_menu` — affordability and the "(short ...)" line through
   `plan_payment`.
6. `hot_reload::DefinitionErrors::economy` + a fourth section in the errors
   panel.
7. `assets/city/economy.ron` — the shipped grant (dirt, cobblestone, planks,
   logs) and a first conversion table: every wood type's log to its planks,
   planks to sticks, cobblestone to stone, stone to bricks, sand to glass,
   raw ore to ingots.

## Done when

- `cargo check` and `cargo test` pass.
- A save with no `stock.ron` opens with the granted materials; one with an
  empty `stock.ron` opens empty.
- A city holding only logs can place a building costing planks, and undoing
  it gives the logs back rather than the planks.
- A table with `A -> B` and `B -> A` in it doesn't hang the game.
