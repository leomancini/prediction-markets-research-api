import "dotenv/config";
import fs from "node:fs";
import express from "express";
import fetch from "node-fetch";
import Anthropic from "@anthropic-ai/sdk";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const app = express();
const port = 3111;

app.use(express.json());

// Initialize the Claude API client (reads ANTHROPIC_API_KEY from .env)
const anthropic = new Anthropic();
const TITLE_MODEL = "claude-haiku-5-5";

// Cache configuration
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
const MAX_MARKETS = 250; // Maximum number of markets to process
const MAX_EVENTS = 150; // Number of Polymarket events to fetch (by volume)
const MAX_SUBMARKETS_PER_EVENT = 2; // Cap options taken from a multi-option event
let cachedMarkets = null;
let cacheTimestamp = null;

// Terms to filter out from market results
const FILTER_TERMS = [
  "bitcoin",
  "btc",
  "ethereum",
  "eth",
  "solana",
  "sol",
  "dogecoin",
  "doge",
  "cardano",
  "ada",
  "polkadot",
  "dot",
  "binance",
  "bnb",
  "chainlink",
  "link",
  "litecoin",
  "ltc",
  "polygon",
  "matic",
  "avalanche",
  "avax",
  "shiba",
  "shib",
  "uniswap",
  "uni",
  "cosmos",
  "atom",
  "algorand",
  "algo",
  "tron",
  "trx",
  "stellar",
  "xlm",
  "monero",
  "xmr",
  "eos",
  "tezos",
  "xtz",
  "dash",
  "zcash",
  "iota",
  "miota",
  "neo",
  "maker",
  "mkr",
  "compound",
  "comp",
  "aave",
  "sushi",
  "yearn",
  "yfi",
  "pancakeswap",
  "cake",
  "ftx",
  "ftt",
  "celsius",
  "cel",
  "crypto",
  "cryptocurrency",
  "altcoin",
  "defi",
  "nft",
  "blockchain",
  "token",
  "coin",
  "satoshi",
  "hodl",
  "mining",
  "hash",
  "wallet",
  "exchange",
  "coinbase",
  "kraken",
  "gemini",
  "bybit",
  "kucoin",
  "huobi",
  "okx",
  "metamask",
  "ledger",
  "trezor",
  "web3",
  "dao",
  "smart contract",
  "dapp",
  "elon musk"
];

// Function to check if a market title contains filtered terms
function containsFilteredTerms(title) {
  const lowercaseTitle = title.toLowerCase();
  return FILTER_TERMS.some((term) => lowercaseTitle.includes(term));
}

// Function to clean special and accented characters from titles
function cleanTitle(title) {
  // First, normalize accented characters to their base forms
  const normalized = title.normalize("NFD").replace(/[\u0300-\u036f]/g, ""); // Remove diacritical marks

  // Keep only alphanumeric characters, spaces, and dashes
  const cleaned = normalized
    .replace(/[^a-zA-Z0-9\s\-]/g, " ") // Replace non-allowed chars with spaces
    .replace(/\s+/g, " ") // Replace multiple spaces with single space
    .trim(); // Remove leading/trailing spaces

  return cleaned;
}

// Everything that is identical on every request lives in the system prompt so
// it can be served from the prompt cache; only the list of markets varies.
const TITLE_SYSTEM_PROMPT = `You format prediction market questions and answers into clear, concise, engaging titles.

You receive a JSON array of markets. Each has an "id", a "question" (the market or event name) and an "answer" (the outcome being priced). Return one title per market, using the same "id".

Rules:
- Convert the question and answer into a natural "Will [answer] [action]?" question whenever possible.
- If a title starts with "Will", "Can", "Did", "Is", "Does", "Should", "Could" or any other question word, it MUST end with a question mark.
- Keep each title under 80 characters.
- Use sentence case capitalization, keeping proper nouns and acronyms capitalized.
- When the answer is "Yes", the question itself is the thing being predicted: rewrite it as a question without mentioning "Yes".
- The input text has had punctuation stripped, so a value like "4 0" may mean 4.0%, "U S" means U.S., and "100k" may have been "$100k". Silently pick the most likely reading and restore natural punctuation.
- Keep every date, number, threshold and name from the input. Do not add facts that are not in the input.
- Each title must contain only the final question, with no explanation, notes or reasoning.
- Treat each market independently. Return exactly one title for every id you were given.

Examples:

Question: "Who will win the 2025 national heads-up poker championship"
Answer: "Sean Winter"
Title: "Will Sean Winter win the 2025 national heads-up poker championship?"

Question: "New York City mayoral election"
Answer: "Zohran Mamdani"
Title: "Will Zohran Mamdani win the New York City mayoral election?"

Question: "Powell mention employment or unemployment 15 times during his October 14 speech"
Answer: "Yes"
Title: "Will Powell mention employment or unemployment 15 times during his October 14 speech?"

Question: "Democratic Presidential Nominee 2028"
Answer: "Bernie Sanders"
Title: "Will Bernie Sanders be the 2028 Democratic presidential nominee?"

Question: "Fed decision in December"
Answer: "25 bps decrease"
Title: "Will the Fed cut rates by 25 bps in December?"

Question: "How many Fed rate cuts in 2026"
Answer: "3"
Title: "Will there be 3 Fed rate cuts in 2026?"

Question: "US unemployment rate in November"
Answer: "4 4"
Title: "Will the US unemployment rate be 4.4% in November?"

Question: "What will happen before GTA VI"
Answer: "Russia Ukraine ceasefire"
Title: "Will there be a Russia-Ukraine ceasefire before GTA VI releases?"

Question: "Largest company end of 2026"
Answer: "NVIDIA"
Title: "Will NVIDIA be the largest company at the end of 2026?"

Question: "Russia x Ukraine ceasefire by March 31"
Answer: "Yes"
Title: "Will Russia and Ukraine agree to a ceasefire by March 31?"`;

const QUESTION_WORDS = [
  "Will",
  "Can",
  "Did",
  "Is",
  "Does",
  "Should",
  "Could",
  "Would",
  "Has",
  "Have",
  "Are",
  "Was",
  "Were"
];

function startsWithQuestionWord(text) {
  return QUESTION_WORDS.some((word) => text.startsWith(word + " "));
}

// Tidy a title Claude returned. Returns null if it is empty.
function finalizeTitle(rawTitle) {
  let title = String(rawTitle ?? "")
    .replace(/\s+/g, " ")
    .trim();
  if (!title) return null;
  // Safety check: add a question mark if the title starts with a question
  // word but doesn't end with one.
  if (startsWithQuestionWord(title) && !title.endsWith("?")) title += "?";
  return title;
}

// A binary market whose original title is already a clean question ("Will X
// happen by Y?") needs no formatting. Only used when stripping punctuation
// would change nothing but the trailing question mark.
function readyMadeTitle(market) {
  const raw = market.rawQuestion;
  if (!raw || market.prediction !== "Yes" || !raw.endsWith("?")) return null;
  if (!startsWithQuestionWord(raw)) return null;
  if (cleanTitle(raw) !== raw.slice(0, -1).trim()) return null;
  return raw;
}

// Format a batch of markets in a single Claude request. Returns an array of
// titles in the same order as `batch`, with null where formatting failed.
// `usage` accumulates token counts across a whole refresh.
async function formatTitleBatch(batch, usage) {
  const items = batch.map((market, id) => ({
    id,
    question: market.question,
    answer: market.prediction
  }));

  try {
    const response = await anthropic.messages.create({
      model: TITLE_MODEL,
      // Room for a full batch of titles plus the model's (low-effort) thinking.
      max_tokens: 4096,
      output_config: {
        effort: "low",
        format: {
          type: "json_schema",
          schema: {
            type: "object",
            properties: {
              titles: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    id: { type: "integer" },
                    title: { type: "string" }
                  },
                  required: ["id", "title"],
                  additionalProperties: false
                }
              }
            },
            required: ["titles"],
            additionalProperties: false
          }
        }
      },
      // The breakpoint sits at the end of the shared prefix; the user message
      // below differs on every request and is deliberately left uncached.
      system: [
        {
          type: "text",
          text: TITLE_SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" }
        }
      ],
      messages: [
        {
          role: "user",
          content: `Format these markets:\n${JSON.stringify(items)}`
        }
      ]
    });

    usage.requests += 1;
    usage.input += response.usage.input_tokens ?? 0;
    usage.output += response.usage.output_tokens ?? 0;
    usage.cacheRead += response.usage.cache_read_input_tokens ?? 0;
    usage.cacheWrite += response.usage.cache_creation_input_tokens ?? 0;

    if (response.stop_reason !== "end_turn") {
      throw new Error(`Claude stopped with "${response.stop_reason}"`);
    }

    const replyText = response.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("");
    const titles = new Array(batch.length).fill(null);
    for (const entry of JSON.parse(replyText).titles ?? []) {
      if (Number.isInteger(entry.id) && entry.id >= 0 && entry.id < batch.length) {
        titles[entry.id] = finalizeTitle(entry.title);
      }
    }
    return titles;
  } catch (error) {
    const isApiError = error instanceof Anthropic.APIError;
    // A refusal, truncation or unparseable reply may be down to one market.
    // Halve the batch and retry so it can't take the rest down with it.
    // API errors (rate limits, outages) are not retried here: the SDK already
    // retried them, and these markets are picked up again on the next refresh.
    if (!isApiError && batch.length > 1) {
      const middle = Math.ceil(batch.length / 2);
      const halves = await Promise.all([
        formatTitleBatch(batch.slice(0, middle), usage),
        formatTitleBatch(batch.slice(middle), usage)
      ]);
      return halves.flat();
    }
    // Log one line per failure. Printing the whole error object dumps the full
    // HTTP response and filled the error log when every request was failing.
    const status = isApiError ? ` (HTTP ${error.status})` : "";
    console.error(
      `Error formatting ${batch.length} title(s), first "${batch[0].question}" / "${batch[0].prediction}"${status}: ${error.message}`
    );
    return batch.map(() => null);
  }
}

// Extract the valid, deduped markets from a single raw event, appending them
// into `markets`. Returns true once MAX_MARKETS has been reached so the caller
// can stop early. Keeping this per-event lets the caller discard each raw page
// as soon as it's parsed instead of holding the whole feed in memory.
function collectMarketsFromEvent(event, markets, seenTitles) {
  const eventMarkets = event.markets || [];
  const isMultiOption = eventMarkets.length > 1;

  // Parse and keep only valid markets for this event.
  let parsed = eventMarkets
    .map((m) => parseMarket(m, event.title, isMultiOption))
    .filter(Boolean);

  // For multi-option events, cap the number of options we keep, taking the
  // highest-volume ones so a single large event can't dominate the feed.
  if (isMultiOption && parsed.length > MAX_SUBMARKETS_PER_EVENT) {
    parsed = parsed
      .sort((a, b) => b.volume - a.volume)
      .slice(0, MAX_SUBMARKETS_PER_EVENT);
  }

  for (const market of parsed) {
    if (markets.length >= MAX_MARKETS) return true;

    const titleKey = `${market.question}-${market.prediction}`;
    if (seenTitles.has(titleKey)) continue;
    seenTitles.add(titleKey);

    markets.push(market);
  }

  return markets.length >= MAX_MARKETS;
}

// Parse a single Polymarket market into our shape, or return null if invalid.
// `eventTitle` is used as the question for multi-option events.
function parseMarket(market, eventTitle, isMultiOption) {
  const outcomes = JSON.parse(market.outcomes || "[]");
  const outcomePrices = JSON.parse(market.outcomePrices || "[]");
  const volume = Math.round(market.volumeNum || parseFloat(market.volume) || 0);

  if (!outcomes.length || !outcomePrices.length || !volume) return null;

  const probability = parseFloat(outcomePrices[0]);
  if (isNaN(probability)) return null;

  // For binary Yes/No markets, use the event title as the question.
  // For multi-option markets, use the event title as the question and the
  // option label (groupItemTitle) as the prediction.
  const question = cleanTitle(eventTitle || market.question || "");
  let prediction;

  if (isMultiOption) {
    // groupItemTitle is the clean per-option label Polymarket uses for both
    // winner-takes-all markets ("Spain") and grouped Yes/No markets
    // ("Troop Withdrawal"). Fall back to the older heuristics if it's absent.
    if (market.groupItemTitle) {
      prediction = cleanTitle(market.groupItemTitle);
    } else {
      const marketQuestion = market.question || "";
      const willMatch = marketQuestion.match(
        /^Will (.+?)(?:\s+win\b|\s+be\b|\s+become\b|\s+qualify\b)/i
      );
      prediction = willMatch ? cleanTitle(willMatch[1]) : cleanTitle(outcomes[0]);
    }
  } else {
    prediction = "Yes";
  }

  if (!question || !prediction) return null;

  // Original, unstripped title of a binary market, kept so a title that is
  // already a well-formed question can be used as-is.
  const rawQuestion = isMultiOption ? null : (eventTitle || "").trim();

  return { question, prediction, probability, volume, rawQuestion };
}

// Function to fetch Polymarket markets via their API.
// The Gamma API caps a single page at 100, so we paginate until we have
// MAX_EVENTS. Each raw page is parsed down to our slim market shape and then
// dropped, so the fat raw event objects (full metadata, images, descriptions)
// are never all held in memory at once and don't survive into Claude processing.
async function scrapePolymarketMarkets() {
  try {
    const PAGE_SIZE = 100;
    const markets = [];
    const seenTitles = new Set();

    for (
      let offset = 0;
      offset < MAX_EVENTS && markets.length < MAX_MARKETS;
      offset += PAGE_SIZE
    ) {
      const limit = Math.min(PAGE_SIZE, MAX_EVENTS - offset);
      const response = await fetch(
        `https://gamma-api.polymarket.com/events?limit=${limit}&offset=${offset}&active=true&closed=false&order=volume&ascending=false`
      );
      let page = await response.json();
      if (!Array.isArray(page) || page.length === 0) break;

      const noMorePages = page.length < limit; // API returned fewer than asked

      let reachedMax = false;
      for (const event of page) {
        if (collectMarketsFromEvent(event, markets, seenTitles)) {
          reachedMax = true;
          break;
        }
      }

      // Drop the raw page's fat event objects before the next fetch so peak
      // memory stays flat regardless of MAX_EVENTS.
      page = null;

      if (reachedMax || noMorePages) break;
    }

    return markets;
  } catch (error) {
    console.error("Error fetching Polymarket markets:", error);
    throw error;
  }
}

// Markets per Claude request. Batching means the fixed instructions are paid
// for once per batch rather than once per market.
const TITLE_BATCH_SIZE = 25;

// Max number of concurrent Claude requests in flight at once.
const CLAUDE_CONCURRENCY = 5;

// Titles depend only on question + prediction, which rarely change between
// refreshes, so each one is formatted once and reused. Persisted to disk so a
// restart or deploy doesn't re-format everything.
const TITLE_CACHE_FILE = new URL("./title-cache.json", import.meta.url);
let titleCache = new Map();
try {
  titleCache = new Map(
    Object.entries(JSON.parse(fs.readFileSync(TITLE_CACHE_FILE, "utf8")))
  );
  console.log(`Loaded ${titleCache.size} cached titles from disk`);
} catch {
  // No cache file yet (or unreadable): start empty.
}

async function saveTitleCache() {
  try {
    await fs.promises.writeFile(
      TITLE_CACHE_FILE,
      JSON.stringify(Object.fromEntries(titleCache))
    );
  } catch (error) {
    console.error(`Could not save title cache: ${error.message}`);
  }
}

// Function to give every market a formatted title, calling Claude only for
// markets that have not been formatted before.
async function processMarketsWithClaude(markets) {
  const marketsToProcess = markets.slice(0, MAX_MARKETS);
  const titles = new Array(marketsToProcess.length).fill(null);
  const pending = []; // indexes that need Claude
  let reused = 0;
  let readyMade = 0;

  marketsToProcess.forEach((market, index) => {
    const cached = titleCache.get(marketKey(market));
    if (cached) {
      titles[index] = cached;
      reused += 1;
      return;
    }
    const ready = readyMadeTitle(market);
    if (ready) {
      titles[index] = ready;
      readyMade += 1;
      return;
    }
    pending.push(index);
  });

  const usage = { requests: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  const batches = [];
  for (let i = 0; i < pending.length; i += TITLE_BATCH_SIZE) {
    batches.push(pending.slice(i, i + TITLE_BATCH_SIZE));
  }

  const runBatch = async (indexes) => {
    const results = await formatTitleBatch(
      indexes.map((index) => marketsToProcess[index]),
      usage
    );
    indexes.forEach((index, i) => {
      titles[index] = results[i];
    });
  };

  // Run the first batch alone: a prompt cache entry can only be read once the
  // request that writes it has responded, so firing everything at once would
  // make every request pay for a cache write.
  if (batches.length > 0) await runBatch(batches[0]);
  for (let i = 1; i < batches.length; i += CLAUDE_CONCURRENCY) {
    await Promise.all(batches.slice(i, i + CLAUDE_CONCURRENCY).map(runBatch));
  }

  // Remember successful titles, and drop ones for markets no longer in the
  // feed so the cache can't grow without bound. Failed titles are not cached,
  // so they are retried on the next refresh.
  const nextCache = new Map();
  let failed = 0;
  const processedMarkets = marketsToProcess.map((market, index) => {
    const title = titles[index];
    if (title) nextCache.set(marketKey(market), title);
    else failed += 1;
    return {
      // Fall back to the combined question and answer if Claude failed.
      title: title ?? `${market.question} - ${market.prediction}`,
      question: market.question,
      prediction: market.prediction,
      probability: market.probability,
      volume: market.volume
    };
  });
  titleCache = nextCache;
  if (pending.length > 0) await saveTitleCache();

  console.log(
    `Titles: ${reused} reused, ${readyMade} already questions, ${pending.length} sent to Claude in ${usage.requests} request(s), ${failed} failed. ` +
      `Tokens: ${usage.input} input, ${usage.output} output, ${usage.cacheRead} cache read, ${usage.cacheWrite} cache write`
  );
  return processedMarkets;
}

// The refresh currently running, if any. Every caller shares it, so a burst of
// requests against an empty or expired cache starts one refresh, not one each.
let refreshInFlight = null;

function refreshMarkets() {
  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const freshMarkets = await scrapePolymarketMarkets();
        cachedMarkets = await processMarketsWithClaude(freshMarkets);
        cacheTimestamp = Date.now();
      } finally {
        refreshInFlight = null;
      }
    })();
  }
  return refreshInFlight;
}

// Function to get cached markets with background refresh
async function getCachedMarkets() {
  // If no cache exists at all, wait for the first fetch
  if (!cachedMarkets) {
    if (!refreshInFlight) {
      console.log(
        "No cache found. Fetching and processing markets for the first time..."
      );
    }
    await refreshMarkets();
    return cachedMarkets;
  }

  // If cache is expired, refresh in background but return stale data immediately
  const isCacheExpired = Date.now() - cacheTimestamp > CACHE_DURATION;
  if (isCacheExpired && !refreshInFlight) {
    const cacheAge = Math.round((Date.now() - cacheTimestamp) / 1000 / 60); // minutes
    console.log(
      `Cache expired (${cacheAge} minutes old). Returning stale data and refreshing in background...`
    );
    refreshMarkets()
      .then(() => console.log("Background cache refresh completed successfully"))
      // Keep the old cache if refresh fails
      .catch((error) => console.error("Background cache refresh failed:", error));
  }

  return cachedMarkets;
}

app.get("/", (req, res) => {
  res.send("Hello world!");
});

// Parse `min`/`max` query params (percentages, 0-100) into a probability
// range (0-1). Invalid or missing values are ignored. Returns { min, max }.
function parseProbabilityRange(query) {
  const toProbability = (raw) => {
    if (raw === undefined) return undefined;
    const pct = parseFloat(raw);
    if (isNaN(pct)) return undefined;
    return Math.min(100, Math.max(0, pct)) / 100;
  };
  return { min: toProbability(query.min), max: toProbability(query.max) };
}

// Helper: return valid, filtered markets (shared by /all, /random, and MCP).
// Optionally constrain to a probability range via { min, max } in 0-1.
async function getFilteredMarkets({ min, max } = {}) {
  const markets = await getCachedMarkets();
  return markets
    .filter(
      (market) =>
        market.probability !== null &&
        market.probability !== undefined &&
        !isNaN(market.probability) &&
        !containsFilteredTerms(market.question) &&
        (min === undefined || market.probability >= min) &&
        (max === undefined || market.probability <= max)
    )
    .slice(0, MAX_MARKETS);
}

// API endpoint to get scraped markets.
// Optional query params: ?min=X&max=Y (percentages) to bound probability.
app.get("/all", async (req, res) => {
  try {
    const validMarkets = await getFilteredMarkets(
      parseProbabilityRange(req.query)
    );
    res.json(validMarkets);
  } catch (error) {
    res.status(500).json({
      error: error.message
    });
  }
});

// Identity key for a market (probability can drift, but question+prediction
// uniquely identifies it — same key used for dedupe during scraping).
function marketKey(market) {
  return `${market.question}-${market.prediction}`;
}

// Minimum probability gap (10 percentage points) between consecutive /random
// results, so we don't show two numerically similar markets back-to-back.
const MIN_PROB_DELTA = 0.1;

// Track the last market returned by /random (key + probability) so we never
// repeat it back-to-back and can favor a meaningfully different probability.
let lastRandom = null;

// API endpoint to get a random market.
// Honors the same ?min=X&max=Y probability filter as /all.
app.get("/random", async (req, res) => {
  try {
    const validMarkets = await getFilteredMarkets(
      parseProbabilityRange(req.query)
    );

    if (validMarkets.length === 0) {
      return res.status(404).json({
        error: "No valid markets found"
      });
    }

    // Narrow the pool relative to the last result, each step only applied when
    // it leaves at least one candidate — so we always return something.
    let candidates = validMarkets;
    if (lastRandom !== null && candidates.length > 1) {
      // 1. Never repeat the exact same market twice in a row.
      const notSame = candidates.filter((m) => marketKey(m) !== lastRandom.key);
      if (notSame.length > 0) candidates = notSame;

      // 2. Prefer a probability at least MIN_PROB_DELTA away from the last one.
      const distinct = candidates.filter(
        (m) => Math.abs(m.probability - lastRandom.probability) >= MIN_PROB_DELTA
      );
      if (distinct.length > 0) candidates = distinct;
    }

    // Get a random market from the candidate array
    const randomIndex = Math.floor(Math.random() * candidates.length);
    const randomMarket = candidates[randomIndex];
    lastRandom = {
      key: marketKey(randomMarket),
      probability: randomMarket.probability
    };

    res.json(randomMarket);
  } catch (error) {
    res.status(500).json({
      error: error.message
    });
  }
});

// Build MCP server exposing the same data as the REST endpoints
function buildMcpServer() {
  const server = new McpServer({
    name: "prediction-markets-research",
    version: "1.0.0"
  });

  server.tool(
    "get_all_markets",
    "Get all cached Polymarket prediction markets (filtered, up to 100).",
    {},
    async () => {
      const markets = await getFilteredMarkets();
      return {
        content: [{ type: "text", text: JSON.stringify(markets, null, 2) }]
      };
    }
  );

  server.tool(
    "get_random_market",
    "Get one random Polymarket prediction market from the filtered set.",
    {},
    async () => {
      const markets = await getFilteredMarkets();
      if (markets.length === 0) {
        return {
          isError: true,
          content: [{ type: "text", text: "No valid markets found" }]
        };
      }
      const random = markets[Math.floor(Math.random() * markets.length)];
      return {
        content: [{ type: "text", text: JSON.stringify(random, null, 2) }]
      };
    }
  );

  return server;
}

app.post("/mcp", async (req, res) => {
  try {
    const server = buildMcpServer();
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined
    });
    res.on("close", () => {
      transport.close();
      server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error("MCP error:", error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: { code: -32603, message: "Internal server error" },
        id: null
      });
    }
  }
});

app.get("/mcp", (req, res) => {
  res.status(405).json({
    jsonrpc: "2.0",
    error: { code: -32000, message: "Method not allowed." },
    id: null
  });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
