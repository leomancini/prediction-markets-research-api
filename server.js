import "dotenv/config";
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
const TITLE_MODEL = "claude-sonnet-5-5";

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

// Function to format a title using Claude
async function formatTitleWithClaude(question, answer) {
  try {
    const response = await anthropic.beta.messages.create({
      model: TITLE_MODEL,
      max_tokens: 1024,
      // Short formatting task: keep thinking and token spend low.
      // The reply is constrained to {"title": "..."} so the model cannot put
      // notes about how it read the input in front of the title.
      output_config: {
        effort: "low",
        format: {
          type: "json_schema",
          schema: {
            type: "object",
            properties: { title: { type: "string" } },
            required: ["title"],
            additionalProperties: false
          }
        }
      },
      // If a safety classifier declines a request, let the API retry it on a
      // fallback model instead of returning nothing.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system:
        "You are a helpful assistant that formats prediction market questions and answers into clear, concise, and engaging titles. Convert the question and answer into a natural 'Will [answer] [action]?' format when possible. IMPORTANT: If the title starts with 'Will', 'Can', 'Did', 'Is', 'Does', 'Should', 'Could', or any other question word, you MUST end it with a question mark (?). Keep them under 80 characters. Use sentence case capitalization. The answer text has had punctuation stripped, so a value like \"4 0\" may mean 4.0%: silently pick the most likely reading. The title field must contain only the final question, with no explanation, notes or reasoning.",
      messages: [
        {
          role: "user",
          content: `Format this prediction market into a clear and engaging question:\nQuestion: "${question}"\nAnswer: "${answer}"\n\nExamples:\nQuestion: "Who will win the 2025 national heads-up poker championship?"\nAnswer: "Sean Winter"\nOutput: "Will Sean Winter win the 2025 national heads-up poker championship?"\n\nQuestion: "New York City mayoral election"\nAnswer: "Zohran Mamdani"\nOutput: "Will Zohran Mamdani win the New York City mayoral election?"\n\nQuestion: "Powell mention employment or unemployment 15+ times during his October 14 speech"\nAnswer: "Yes"\nOutput: "Will Powell mention employment or unemployment 15 times during his October 14 speech?"\n\nRespond with just the formatted question.`
        }
      ]
    });

    if (response.stop_reason === "refusal") {
      throw new Error("Claude declined to format this title");
    }

    const replyText = response.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("");
    let formattedTitle = String(JSON.parse(replyText).title ?? "")
      .replace(/\s+/g, " ")
      .trim();
    if (!formattedTitle) {
      throw new Error("Claude returned no text");
    }

    // Safety check: Add question mark if title starts with a question word but doesn't end with one
    const questionWords = [
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
    const startsWithQuestionWord = questionWords.some((word) =>
      formattedTitle.startsWith(word + " ")
    );
    if (startsWithQuestionWord && !formattedTitle.endsWith("?")) {
      formattedTitle += "?";
    }

    return formattedTitle;
  } catch (error) {
    // Log one line per failure. Printing the whole error object dumps the full
    // HTTP response and filled the error log when every request was failing.
    const status = error instanceof Anthropic.APIError ? ` (HTTP ${error.status})` : "";
    console.error(
      `Error formatting question "${question}" with answer "${answer}"${status}: ${error.message}`
    );
    // Return combined question and answer if Claude fails
    return `${question} - ${answer}`;
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

  return { question, prediction, probability, volume };
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

// Max number of concurrent Claude requests in flight at once. Bounds peak memory
// (each in-flight request holds a TLS socket + buffers) and eases Anthropic rate limits.
const CLAUDE_CONCURRENCY = 20;

// Function to process markets with Claude formatting
async function processMarketsWithClaude(markets) {
  // Limit to MAX_MARKETS
  const marketsToProcess = markets.slice(0, MAX_MARKETS);
  console.log(`Processing ${marketsToProcess.length} markets with Claude...`);

  // Process in bounded batches instead of firing all requests at once, so peak
  // memory stays flat regardless of MAX_MARKETS.
  const processedMarkets = [];
  for (let i = 0; i < marketsToProcess.length; i += CLAUDE_CONCURRENCY) {
    const batch = marketsToProcess.slice(i, i + CLAUDE_CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map(async (market) => {
        const formattedTitle = await formatTitleWithClaude(
          market.question,
          market.prediction
        );
        return {
          title: formattedTitle,
          question: market.question,
          prediction: market.prediction,
          probability: market.probability,
          volume: market.volume
        };
      })
    );
    processedMarkets.push(...batchResults);
  }

  console.log(
    `Finished processing ${processedMarkets.length} markets with Claude`
  );
  return processedMarkets;
}

// Function to get cached markets with background refresh
async function getCachedMarkets() {
  const now = Date.now();
  const isCacheExpired =
    !cacheTimestamp || now - cacheTimestamp > CACHE_DURATION;

  // If no cache exists at all, fetch synchronously (first time)
  if (!cachedMarkets) {
    console.log(
      "No cache found. Fetching and processing markets for the first time..."
    );
    const markets = await scrapePolymarketMarkets();
    cachedMarkets = await processMarketsWithClaude(markets);
    cacheTimestamp = now;
    console.log("Initial cache populated successfully");

    return cachedMarkets;
  }

  // If cache is expired, refresh in background but return stale data immediately
  if (isCacheExpired) {
    const cacheAge = Math.round((now - cacheTimestamp) / 1000 / 60); // minutes
    console.log(
      `Cache expired (${cacheAge} minutes old). Returning stale data and refreshing in background...`
    );

    // Refresh cache in background (don't await)
    refreshCacheInBackground();

    // Return stale data immediately
    return cachedMarkets;
  }

  // Cache is fresh
  return cachedMarkets;
}

// Background cache refresh function
async function refreshCacheInBackground() {
  try {
    console.log("Starting background cache refresh...");
    const freshMarkets = await scrapePolymarketMarkets();
    const processedMarkets = await processMarketsWithClaude(freshMarkets);
    cachedMarkets = processedMarkets;
    cacheTimestamp = Date.now();
    console.log("Background cache refresh completed successfully");
  } catch (error) {
    console.error("Background cache refresh failed:", error);
    // Keep the old cache if refresh fails
  }
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
