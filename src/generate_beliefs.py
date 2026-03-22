"""Generate 10,000 diverse belief statements using GPT-4.1."""

import json
import os
import asyncio
import random
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 50 diverse topic categories to ensure breadth
TOPICS = [
    "politics and governance", "economics and wealth inequality",
    "religion and spirituality", "science and technology",
    "education and learning", "healthcare and medicine",
    "environment and climate", "social justice and equality",
    "family and relationships", "gender and sexuality",
    "immigration and borders", "criminal justice and policing",
    "military and warfare", "media and journalism",
    "art and culture", "sports and competition",
    "food and agriculture", "housing and urban planning",
    "transportation and infrastructure", "privacy and surveillance",
    "artificial intelligence and automation", "space exploration",
    "mental health and wellbeing", "drugs and substance policy",
    "animal rights and welfare", "gun rights and control",
    "free speech and censorship", "death and end of life",
    "tradition versus progress", "nationalism and globalism",
    "work ethic and labor", "wealth and materialism",
    "democracy and authoritarianism", "human nature and morality",
    "parenting and child-rearing", "aging and elderly care",
    "beauty and body image", "conspiracy theories and trust",
    "luck versus effort", "punishment versus rehabilitation",
    "individual rights versus collective good", "risk and safety",
    "privacy versus transparency", "competition versus cooperation",
    "nature versus nurture", "freedom versus security",
    "local versus global priorities", "innovation versus stability",
    "meritocracy and fairness", "digital life and social media",
]

# Modifiers for diversity within topics
ANGLES = [
    "controversial", "mainstream", "nuanced", "extreme",
    "philosophical", "practical", "personal", "societal",
    "optimistic", "pessimistic", "traditional", "progressive",
    "libertarian", "communitarian", "scientific", "spiritual",
    "economic", "ethical", "cultural", "pragmatic",
]


async def generate_batch(topic: str, angle: str, count: int, batch_id: int) -> list[str]:
    """Generate a batch of belief statements."""
    prompt = f"""Generate exactly {count} diverse belief statements about "{topic}" from a {angle} perspective.

Rules:
- Each belief should be a clear, first-person statement (e.g., "I believe that...")
- Beliefs should be specific enough to agree or disagree with
- Cover different sub-aspects of the topic
- Include both common and uncommon viewpoints
- No duplicates or near-duplicates
- One belief per line, numbered 1-{count}
- Do NOT include explanations, just the belief statements"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=4000,
        )
        text = response.choices[0].message.content
        beliefs = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering
            for i in range(count + 1):
                line = line.lstrip(f"{i}.")
                line = line.lstrip(f"{i})")
            line = line.strip().strip("-").strip("*").strip()
            if len(line) > 10:  # Filter very short/empty lines
                beliefs.append(line)
        return beliefs
    except Exception as e:
        print(f"Error in batch {batch_id}: {e}")
        return []


async def generate_all_beliefs(target: int = 10000) -> list[str]:
    """Generate target number of diverse beliefs."""
    # Create all topic-angle combinations
    combos = [(t, a) for t in TOPICS for a in ANGLES]
    random.shuffle(combos)

    # Calculate beliefs per batch (aim for ~10 per batch, ~1000 batches)
    beliefs_per_batch = 10
    num_batches = (target // beliefs_per_batch) + 100  # Extra buffer

    tasks = []
    for i in range(min(num_batches, len(combos))):
        topic, angle = combos[i % len(combos)]
        tasks.append(generate_batch(topic, angle, beliefs_per_batch, i))

    # Run in parallel with semaphore to avoid rate limits
    sem = asyncio.Semaphore(50)

    async def limited(coro):
        async with sem:
            return await coro

    print(f"Generating beliefs with {len(tasks)} API calls...")
    results = await asyncio.gather(*[limited(t) for t in tasks])

    all_beliefs = []
    for batch in results:
        all_beliefs.extend(batch)

    # Deduplicate (exact match)
    seen = set()
    unique = []
    for b in all_beliefs:
        b_lower = b.lower().strip()
        if b_lower not in seen:
            seen.add(b_lower)
            unique.append(b)

    print(f"Generated {len(unique)} unique beliefs from {len(all_beliefs)} total")
    return unique[:target]


async def main():
    # Also load existing beliefs
    with open("datasets/belief_statements.json") as f:
        existing = json.load(f)

    existing_beliefs = []
    for item in existing:
        for p in item.get("perspectives", []):
            existing_beliefs.append(p)

    print(f"Loaded {len(existing_beliefs)} existing belief perspectives")

    # Generate new beliefs
    generated = await generate_all_beliefs(target=10000 - len(existing_beliefs))

    # Combine
    all_beliefs = existing_beliefs + generated

    # Final dedup
    seen = set()
    unique = []
    for b in all_beliefs:
        key = b.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(b)

    # Trim to 10,000
    unique = unique[:10000]
    print(f"Final belief count: {len(unique)}")

    # Save
    with open("results/beliefs_10k.json", "w") as f:
        json.dump(unique, f, indent=2)

    # Print some stats
    print(f"\nSample beliefs:")
    for b in random.sample(unique, min(10, len(unique))):
        print(f"  - {b}")


if __name__ == "__main__":
    asyncio.run(main())
