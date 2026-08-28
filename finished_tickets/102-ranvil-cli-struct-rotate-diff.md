# 102 - ranvil-cli: `struct rotate`, `struct diff`

## Scope

- **`ranvil-cli struct rotate <file.nbt> --by 90|180|270 --out <file2.nbt>
  [--force]`** — a direct wrapper around `blueprint::rotate_blueprint`
  (ticket 038's function, unchanged) with `Rotation`'s existing variants.
  A `RotationError::UnrotatableProperty` (a palette entry whose properties
  `rotate_blueprint` doesn't know how to rewrite) is `CliError::Data`
  naming the offending block name and property, matching what the error
  variant already carries — not a generic failure message.
- **`ranvil-cli struct diff <a.nbt> <b.nbt>`** — reads both structure files;
  refuses (`CliError::Usage`) if their `size`s differ, since a per-position
  diff needs a shared coordinate space (comparing two differently-sized
  buildings is a `struct info a.nbt` and `struct info b.nbt` side by side,
  not this command's job). Otherwise walks every position and reports where
  `block_at` disagrees: `text`/`compact` a count plus a capped sample list
  (same capping convention as `scan`'s `--limit`, default a few hundred,
  `--limit` to raise it); `json` the full list, each entry `{"pos": [x,y,z],
  "a": "...", "b": "..."}`. Properties are part of the comparison here
  (unlike `scan`/`replace`'s name-only matching) — a diff's whole purpose is
  catching exactly the kind of change name-only matching would hide, like a
  rotated door nobody noticed.

## Done when

- `struct rotate --by 180 --by 180` (rotate twice) on a structure equals
  the original — verified with `struct diff` reporting zero differences.
- `struct rotate` on a structure containing a property
  `rotate_blueprint` can't handle exits with that error's own message,
  unchanged from what the function already reports internally.
- `struct diff` between a file and a copy of itself with one block changed
  reports exactly one difference, correct position and before/after values.
- `struct diff` between differently-sized files exits 2 rather than
  attempting a partial comparison.
- `cargo check` and `cargo test` pass.
