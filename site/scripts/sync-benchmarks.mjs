import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

// The website reads exactly the committed synthetic receipts, never fabricated numbers.
const source = fileURLToPath(new URL('../../benchmarks/', import.meta.url));
const destination = fileURLToPath(new URL('../public/benchmarks/', import.meta.url));
await mkdir(destination, { recursive: true });
for (const mode of ['stream', 'resident']) {
  const name = 'build-cpu-' + mode + '.json';
  const content = await readFile(join(source, name), 'utf8');
  const data = JSON.parse(content);
  if (data.mode !== mode || data.device !== 'cpu' || data.provenance !== 'MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED') {
    throw new Error('Refusing incompatible or unverified benchmark receipt: ' + name);
  }
  await writeFile(join(destination, name), content);
}
console.log('Synced verified CPU synthetic receipts to public/benchmarks');
