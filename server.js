import express from "express";
import fetch from "node-fetch";
import * as cheerio from "cheerio";

const app = express();
const port = 3111;

// Cache configuration
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
let cachedMarkets = null;
let cacheTimestamp = null;

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
      let title = titleElement.text().trim();

      // Clean up title format - remove " by...?" for multi-option markets
      const cleanTitle = title.replace(/ by\.\.\.\?$/g, "");

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

          const name = nameEl.text().trim();
          const probability = probabilityEl.text().trim();

          if (name && probability && title && volume) {
            // Use "by" format only if original title contained " by...?"
            const fullTitle = title.includes(" by...?")
              ? `${cleanTitle} by ${name}`
              : `${title} - ${name}`;

            // Only add if we haven't seen this title before
            if (!seenTitles.has(fullTitle)) {
              seenTitles.add(fullTitle);
              const decimalPercent =
                parseInt(probability.replace("%", "")) / 100;
              markets.push({
                title: fullTitle,
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
                title,
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

// Function to get cached markets with background refresh
async function getCachedMarkets() {
  const now = Date.now();
  const isCacheExpired =
    !cacheTimestamp || now - cacheTimestamp > CACHE_DURATION;

  // If no cache exists at all, fetch synchronously (first time)
  if (!cachedMarkets) {
    cachedMarkets = await scrapePolymarketMarkets();
    cacheTimestamp = now;

    return cachedMarkets;
  }

  // If cache is expired, refresh in background but return stale data immediately
  if (isCacheExpired) {
    // Refresh cache in background (don't await)
    refreshCacheInBackground();

    // Return stale data immediately
    return cachedMarkets;
  }

  return cachedMarkets;
}

// Background cache refresh function
async function refreshCacheInBackground() {
  try {
    const freshMarkets = await scrapePolymarketMarkets();
    cachedMarkets = freshMarkets;
    cacheTimestamp = Date.now();
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
    const validMarkets = markets.filter(
      (market) =>
        market.probability !== null &&
        market.probability !== undefined &&
        !isNaN(market.probability)
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
    const validMarkets = markets.filter(
      (market) =>
        market.probability !== null &&
        market.probability !== undefined &&
        !isNaN(market.probability)
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
