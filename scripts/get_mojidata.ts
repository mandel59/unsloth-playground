// Use deno to download moji.db
// deno run --allow-write=moji.db --unstable-raw-imports scripts/get_mojidata.ts
import mojidata from "@mandel59/mojidata/dist/moji.db" with { type: "bytes" };
import { writeFile } from "node:fs/promises";
await writeFile("moji.db", mojidata);
