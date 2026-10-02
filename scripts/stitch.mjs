// iso-stl stitcher — composites a tile grid into one map PNG (+ a small preview).
// Same job as tools/iso-stl/stitch.mjs: lay tile_C_R.png back into column/row order.
//
//   npm install sharp
//   node scripts/stitch.mjs --src ./styled_tiles --out region.png
//
// If styled_tiles/manifest.json exists and has { "grid": N }, the canvas is N x N.
// Otherwise the canvas is sized from the tile_C_R.png names in --src.
import sharp from 'sharp'
import fs from 'node:fs'
import path from 'node:path'

const arg = (k, d) => {
  const i = process.argv.indexOf('--' + k)
  return i > -1 ? process.argv[i + 1] : d
}
const SRC = arg('src', 'styled_tiles')
const OUTP = arg('out', 'map.png')
const CELL = parseInt(arg('cell', '1024'), 10)

function tilesIn(dir) {
  const found = []
  for (const name of fs.readdirSync(dir)) {
    const m = /^tile_(\d+)_(\d+)\.png$/.exec(name)
    if (m) found.push({ col: Number(m[1]), row: Number(m[2]), name })
  }
  if (!found.length) throw new Error(`no tile_C_R.png files in ${dir}`)
  return found
}

const named = tilesIn(SRC)
let cols = Math.max(...named.map((t) => t.col)) + 1
let rows = Math.max(...named.map((t) => t.row)) + 1
const manifestPath = path.join(SRC, 'manifest.json')
if (fs.existsSync(manifestPath)) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  if (Number.isInteger(manifest.grid) && manifest.grid > 0) {
    cols = manifest.grid
    rows = manifest.grid
  }
}

const byName = new Map(named.map((t) => [t.name, t]))
const composites = []
let missing = 0
for (let row = 0; row < rows; row++) {
  for (let col = 0; col < cols; col++) {
    const name = `tile_${col}_${row}.png`
    if (!byName.has(name)) { missing++; continue }
    const buf = await sharp(path.join(SRC, name)).resize(CELL, CELL, { fit: 'fill' }).png().toBuffer()
    composites.push({ input: buf, left: col * CELL, top: row * CELL })
  }
}
if (missing) console.log(`${missing} tiles missing — holes left dark`)

await sharp({
  create: { width: cols * CELL, height: rows * CELL, channels: 3, background: { r: 16, g: 16, b: 20 } },
})
  .composite(composites)
  .png()
  .toFile(OUTP)

const prev = OUTP.replace(/\.png$/, '-preview.png')
await sharp(OUTP).resize(1024, 1024, { fit: 'inside' }).png().toFile(prev)
console.log(`stitched ${cols}x${rows} -> ${OUTP} (+ ${prev})`)
