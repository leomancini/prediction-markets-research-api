import "dotenv/config";
import express from "express";
import fetch from "node-fetch";
import * as cheerio from "cheerio";
import OpenAI from "openai";

const app = express();
const port = 3111;

// Initialize OpenAI
const openai = new OpenAI();

// Cache configuration
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
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
            "You are a helpful assistant that formats prediction market questions and answers into clear, concise, and engaging titles. Convert the question and answer into a natural 'Will [answer] [action]?' format when possible. Always end with a question mark if the title sounds like a question (e.g., starts with 'Will', 'Can', 'Did', 'Is', 'Does', etc.). Keep them under 80 characters. Use sentence case capitalization."
        },
        {
          role: "user",
          content: `Format this prediction market into a clear and engaging question:\nQuestion: "${question}"\nAnswer: "${answer}"\n\nExamples:\nQuestion: "Who will win the 2025 national heads-up poker championship?"\nAnswer: "Sean Winter"\nOutput: "Will Sean Winter win the 2025 national heads-up poker championship?"\n\nQuestion: "New York City mayoral election"\nAnswer: "Zohran Mamdani"\nOutput: "Will Zohran Mamdani win the New York City mayoral election?"\n\nQuestion: "Powell mention employment or unemployment 15+ times during his October 14 speech"\nAnswer: "Yes"\nOutput: "Will Powell mention employment or unemployment 15 times during his October 14 speech?"\n\nRespond with just the formatted question, no quotes around it.`
        }
      ],
      max_completion_tokens: 100
    });

    return response.choices[0].message.content.trim();
  } catch (error) {
    console.error(
      `Error formatting question "${question}" with answer "${answer}":`,
      error
    );
    // Return combined question and answer if GPT fails
    return `${question} - ${answer}`;
  }
}

// Function to scrape Polymarket markets
async function scrapePolymarketMarkets() {
  try {
    const response = await fetch("https://polymarket.com/", {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
      }
    });

    const html = await response.text();
    const $ = cheerio.load(html);
    const markets = [];
    const seenTitles = new Set(); // Track titles we've already processed

    // Find all market cards - they have this distinctive class structure
    $("div.transition.rounded-md.shadow-md").each((index, element) => {
      const $card = $(element);

      // Extract market title
      const titleElement = $card.find("p.text-sm.font-semibold").first();
      let title = cleanTitle(titleElement.text().trim());

      // Clean up title format - remove " by...?" for multi-option markets
      const cleanedTitle = title.replace(/ by\.\.\.\?$/g, "");

      // Extract market link
      const linkElement = $card.find('a[href^="/event/"]').first();
      const link = linkElement.attr("href");

      // Extract market image
      const imgElement = $card.find("img").first();
      const image =
        imgElement.attr("src") || imgElement.attr("srcset")?.split(" ")[0];

      // Extract volume information
      const volumeElement = $card.find('p:contains("Vol.")');
      const volumeText = volumeElement.text().trim();

      // Convert volume to integer (e.g., "$16m Vol." -> 16000000)
      let volume = 0;
      if (volumeText) {
        const match = volumeText.match(/\$?([0-9.]+)([kmb]?)/i);
        if (match) {
          const number = parseFloat(match[1]);
          const unit = match[2].toLowerCase();

          switch (unit) {
            case "k":
              volume = Math.round(number * 1000);
              break;
            case "m":
              volume = Math.round(number * 1000000);
              break;
            case "b":
              volume = Math.round(number * 1000000000);
              break;
            default:
              volume = Math.round(number);
          }
        }
      }

      // Extract betting options with probabilities

      // Check for multi-option markets (like "Democratic Presidential Nominee 2028")
      const multiOptionElements = $card.find(
        "div.flex.justify-between.items-center.gap-4.w-full.h-fit.shrink-0"
      );

      if (multiOptionElements.length > 0) {
        // For multi-option markets, create separate entries for each option (max 5)
        multiOptionElements.slice(0, 5).each((i, optionEl) => {
          const $option = $(optionEl);
          const nameEl = $option.find("p.line-clamp-1.text-\\[13px\\]").first();
          const probabilityEl = $option
            .find("p.font-semibold.text-text-primary.mr-1")
            .first();

          const name = cleanTitle(nameEl.text().trim());
          const probability = probabilityEl.text().trim();

          if (name && probability && title && volume) {
            // Use "by" format only if original title contained " by...?"
            const fullTitle = title.includes(" by...?")
              ? `${cleanedTitle} by ${name}`
              : `${title} - ${name}`;

            // Only add if we haven't seen this title before
            if (!seenTitles.has(fullTitle)) {
              seenTitles.add(fullTitle);
              const decimalPercent =
                parseInt(probability.replace("%", "")) / 100;
              markets.push({
                question: cleanedTitle,
                prediction: name,
                probability: decimalPercent,
                volume
              });
            }
          }
        });
      } else {
        // Check for binary prediction markets (like "Russia x Ukraine ceasefire in 2025?")
        const chanceElement = $card
          .find("p.font-medium.text-\\[16px\\].text-center")
          .first();
        const chanceTextElement = $card.find('p:contains("chance")').first();

        if (chanceElement.length && chanceTextElement.length) {
          const probability = chanceElement.text().trim();

          // Only add markets that have valid data
          if (title && volume && probability) {
            // Only add if we haven't seen this title before
            if (!seenTitles.has(title)) {
              seenTitles.add(title);
              const decimalPercent =
                parseInt(probability.replace("%", "")) / 100;
              markets.push({
                question: title,
                prediction: "Yes",
                probability: decimalPercent,
                volume
              });
            }
          }
        }
      }
    });

    return markets;
  } catch (error) {
    console.error("Error scraping Polymarket:", error);
    throw error;
  }
}

// Function to process markets with GPT formatting
async function processMarketsWithGPT(markets) {
  console.log(`Processing ${markets.length} markets with GPT...`);

  // Process all titles in parallel with Promise.all
  const processedMarkets = await Promise.all(
    markets.map(async (market) => {
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
    res.json(validMarkets);
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
    const validMarkets = markets.filter(
      (market) =>
        market.probability !== null &&
        market.probability !== undefined &&
        !isNaN(market.probability) &&
        !containsFilteredTerms(market.question)
    );

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

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
