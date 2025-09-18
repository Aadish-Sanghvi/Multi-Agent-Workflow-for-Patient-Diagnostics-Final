#normal scraping without llm filtering
import asyncio
from typing import List, Dict
from crawl4ai import AsyncWebCrawler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from srapped_raw_data_structured_raw_data import *
import re
import pandas as pd

# Read your raw file


class WebScrapingAgent:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    @staticmethod
    def structure_to_rawdata(all_raw_contents: list, output_csv: str = "pmc_results.csv"):
        # Combine all raw contents
        combined_raw = "\n".join(all_raw_contents)
        # Regex to match PMC IDs like PMC12345678
        pmc_ids = re.findall(r'PMC\d+', combined_raw)
        pmc_ids = list(set(pmc_ids))
        data = [{"pmcid": pmc, "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"} for pmc in pmc_ids]
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Extracted {len(pmc_ids)} PMC IDs. Saved to {output_csv}")

    async def scrape_page(self, url: str) -> Dict:
        print(f"Scraping: {url}")
        async with AsyncWebCrawler(verbose=True) as crawler:
            crawler.browser_config = {
                "browser_type": "chromium",
                "headless": True,
                "viewport": {"width": 1920, "height": 1080},
            }
            result = await crawler.arun(
                url=url,
                crawl_options={
                    "include_tags": ["p", "article", "div", "section", "h1", "h2", "h3"],
                    "exclude_tags": ["script", "style", "nav", "footer", "header", "aside"],
                }
            )
            if result.success:
                raw_text = result.markdown
                chunks = self.text_splitter.split_text(raw_text) if raw_text else []
                return {
                    "url": url,
                    "title": result.metadata.get("title", "Unknown Title") if result.metadata else "Unknown Title",
                    "raw_content": raw_text,
                    "filtered_chunks": chunks,
                    "success": True
                }
            else:
                return {
                    "url": url,
                    "title": "Scraping Failed",
                    "raw_content": "",
                    "filtered_chunks": [],
                    "success": False,
                    "error": result.error_message
                }
    
    def run(self, url: str) -> Dict:
        return asyncio.run(self.scrape_page(url))

# Example Usage: Loop through pages 1-10
async def main():
    agent = WebScrapingAgent()
    base_url = "https://pmc.ncbi.nlm.nih.gov/search/?term=chronic+cough&sort=relevance&page={page}&ac=no"
    all_raw_contents = []
    for page_number in range(1, 3):  # Pages 1 to 10
        url = base_url.format(page=page_number)
        result = await agent.scrape_page(url)
        print(f"\n--- Page {page_number} ---")
        print(result["raw_content"])
        raw_content = result["raw_content"]
        all_raw_contents.append(raw_content)
    agent.structure_to_rawdata(all_raw_contents)
    await process_urls_from_csv("pmc_results.csv", "scraped_articles.csv")

if __name__ == "__main__":
    asyncio.run(main())

