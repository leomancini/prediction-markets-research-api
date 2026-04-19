import "dotenv/config";
import express from "express";
import fetch from "node-fetch";
import OpenAI from "openai";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const app = express();
const port = 3111;

// Initialize OpenAI
const openai = new OpenAI();

// Cache configuration
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
const MAX_MARKETS = 100; // Maximum number of markets to process
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

// Function to format a title using GPT
async function formatTitleWithGPT(question, answer) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4.1-nano",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant that formats prediction market questions and answers into clear, concise, and engaging titles. Convert the question and answer into a natural 'Will [answer] [action]?' format when possible. IMPORTANT: If the title starts with 'Will', 'Can', 'Did', 'Is', 'Does', 'Should', 'Could', or any other question word, you MUST end it with a question mark (?). Keep them under 80 characters. Use sentence case capitalization."
        },
        {
          role: "user",
          content: `Format this prediction market into a clear and engaging question:\nQuestion: "${question}"\nAnswer: "${answer}"\n\nExamples:\nQuestion: "Who will win the 2025 national heads-up poker championship?"\nAnswer: "Sean Winter"\nOutput: "Will Sean Winter win the 2025 national heads-up poker championship?"\n\nQuestion: "New York City mayoral election"\nAnswer: "Zohran Mamdani"\nOutput: "Will Zohran Mamdani win the New York City mayoral election?"\n\nQuestion: "Powell mention employment or unemployment 15+ times during his October 14 speech"\nAnswer: "Yes"\nOutput: "Will Powell mention employment or unemployment 15 times during his October 14 speech?"\n\nRespond with just the formatted question, no quotes around it.`
        }
      ],
      max_completion_tokens: 100
    });

    let formattedTitle = response.choices[0].message.content.trim();

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
    console.error(
      `Error formatting question "${question}" with answer "${answer}":`,
      error
    );
    // Return combined question and answer if GPT fails
    return `${question} - ${answer}`;
  }
}

// Function to fetch Polymarket markets via their API
async function scrapePolymarketMarkets() {
  try {
    const response = await fetch(
      "https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&order=volume&ascending=false"
    );

    const events = await response.json();
    const markets = [];
    const seenTitles = new Set();

    for (const event of events) {
      if (markets.length >= MAX_MARKETS) break;

      const eventMarkets = event.markets || [];
      const isMultiOption = eventMarkets.length > 1;

      for (const market of eventMarkets) {
        if (markets.length >= MAX_MARKETS) break;

        const outcomes = JSON.parse(market.outcomes || "[]");
        const outcomePrices = JSON.parse(market.outcomePrices || "[]");
        const volume = Math.round(market.volumeNum || parseFloat(market.volume) || 0);

        if (!outcomes.length || !outcomePrices.length || !volume) continue;

        const probability = parseFloat(outcomePrices[0]);
        if (isNaN(probability)) continue;

        // For binary Yes/No markets, use the event title as the question
        // For multi-option markets, use event title as question and extract prediction from market question
        let question = cleanTitle(event.title || market.question || "");
        let prediction;

        if (isMultiOption) {
          // Extract the specific option from the market question
          // e.g., "Will Stephen A. Smith win the 2028 Democratic presidential nomination?" → "Stephen A. Smith"
          const marketQuestion = market.question || "";
          const willMatch = marketQuestion.match(/^Will (.+?)(?:\s+win\b|\s+be\b|\s+become\b|\s+qualify\b)/i);
          prediction = willMatch ? cleanTitle(willMatch[1]) : cleanTitle(outcomes[0]);
        } else {
          prediction = "Yes";
        }

        if (!question || !prediction) continue;

        const titleKey = `${question}-${prediction}`;
        if (seenTitles.has(titleKey)) continue;
        seenTitles.add(titleKey);

        markets.push({
          question,
          prediction,
          probability,
          volume
        });
      }
    }

    return markets;
  } catch (error) {
    console.error("Error fetching Polymarket markets:", error);
    throw error;
  }
}

// Function to process markets with GPT formatting
async function processMarketsWithGPT(markets) {
  // Limit to MAX_MARKETS
  const marketsToProcess = markets.slice(0, MAX_MARKETS);
  console.log(`Processing ${marketsToProcess.length} markets with GPT...`);

  // Process all titles in parallel with Promise.all
  const processedMarkets = await Promise.all(
    marketsToProcess.map(async (market) => {
      const formattedTitle = await formatTitleWithGPT(
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

  console.log(
    `Finished processing ${processedMarkets.length} markets with GPT`
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
    cachedMarkets = await processMarketsWithGPT(markets);
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
    const processedMarkets = await processMarketsWithGPT(freshMarkets);
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

// API endpoint to get scraped markets
app.get("/all", async (req, res) => {
  try {
    const markets = await getCachedMarkets();
    // Filter out markets where probability is null, undefined, or NaN
    // Also filter out markets containing filtered terms (check question)
    const validMarkets = markets.filter(
      (market) =>
        market.probability !== null &&
        market.probability !== undefined &&
        !isNaN(market.probability) &&
        !containsFilteredTerms(market.question)
    );
    // Limit to MAX_MARKETS
    res.json(validMarkets.slice(0, MAX_MARKETS));
  } catch (error) {
    res.status(500).json({
      error: error.message
    });
  }
});

// API endpoint to get a random market
app.get("/random", async (req, res) => {
  try {
    const markets = await getCachedMarkets();
    // Filter out markets where probability is null, undefined, or NaN
    // Also filter out markets containing filtered terms (check question)
    const validMarkets = markets
      .filter(
        (market) =>
          market.probability !== null &&
          market.probability !== undefined &&
          !isNaN(market.probability) &&
          !containsFilteredTerms(market.question)
      )
      .slice(0, MAX_MARKETS); // Limit to MAX_MARKETS

    if (validMarkets.length === 0) {
      return res.status(404).json({
        error: "No valid markets found"
      });
    }

    // Get a random market from the filtered array
    const randomIndex = Math.floor(Math.random() * validMarkets.length);
    const randomMarket = validMarkets[randomIndex];

    res.json(randomMarket);
  } catch (error) {
    res.status(500).json({
      error: error.message
    });
  }
});

// Helper: return valid, filtered markets (same logic as /all and /random)
async function getFilteredMarkets() {
  const markets = await getCachedMarkets();
  return markets
    .filter(
      (market) =>
        market.probability !== null &&
        market.probability !== undefined &&
        !isNaN(market.probability) &&
        !containsFilteredTerms(market.question)
    )
    .slice(0, MAX_MARKETS);
}

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

// MCP endpoint (stateless Streamable HTTP transport)
app.post("/mcp", express.json(), async (req, res) => {
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
    console.error("MCP request error:", error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: { code: -32603, message: "Internal server error" },
        id: null
      });
    }
  }
});

// Reject GET/DELETE on /mcp in stateless mode with a proper JSON-RPC error
app.get("/mcp", (req, res) => {
  res.status(405).json({
    jsonrpc: "2.0",
    error: { code: -32000, message: "Method not allowed." },
    id: null
  });
});
app.delete("/mcp", (req, res) => {
  res.status(405).json({
    jsonrpc: "2.0",
    error: { code: -32000, message: "Method not allowed." },
    id: null
  });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
