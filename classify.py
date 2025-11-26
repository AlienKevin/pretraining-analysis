import os
import json
import glob
import csv
import asyncio
from openai import AsyncOpenAI
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

TAXONOMY = [
    {"name": "News", "example": "PHILADELPHIA -- A nurse cleaning motorists had hair removed from some of their horses he wanted to take in a wagon run on Wednesday afternoon.\n\nThe nurse, who was off shift for money, stopped to retrieve a horse he had removed off Umbridge Road and 30th Street NW.\n\nThe horse, seized by the Suffolk County sheriff, was Walter el Callie. Authorities said Callie's family does not believe he was responsible for the horse's theft.\n\nJust"},
    {"name": "QA", "example": "Question: A 50 year-old man had been off work because of intermittent pain in the right knee for the past 3 months. There is no knee swelling or discoloration on physical examination. Which of the following is the most likely diagnosis?\nA. Ankylosing spondylitis\nB. Osteoarthritis of knee\nC. Rheumatoid arthritis (RA)\nD. Incomplete fracture of os coxa\nAnswer: C"},
    {"name": "Code", "example": "def while_loop_conditional(arr):\n    \"\"\"\n    Complete the while loop and conditional statement to print the elements in the array that satisfy the condition.\n\n    Hints:\n        * Use a while loop to iterate over the elements of the array\n        * Use the if-else statement to check if an element satisfies the condition\n        * Use the enumerate function to get the index of each element in the array\n\n    Input:\n        arr: an array of integers\n\n    Output:\n        The elements in"},
    {"name": "Advertisement", "example": "Download E-health junior farming productivity a simulation and trading models in maize\nto Get Industry Causes directly virtual without book one views you the proof authors commonly also to a Joint, differing your scoring text. An programming implementation for Central algorithm. download E-health junior farming productivity a simulation and trading models in you browse testing for and we'll mechanize it proposed right. And are developed that we can know them with energy for a nonsurgical email, and the important students adjustments, Xenon recommanded based. The"},
    {"name": "Social Media", "example": "Thread: Shit camera's! Botanist!\nAdam Lucas\nYeah, it looks pretty good from my phone, and I think Kermit can use a video.\nOriginally Posted by: lockjaw\nThat was actually a question we asked Kermit. He had the old running system case and short sleeve shirt laying around, and I thought it would be better to film the hands and feet than to wait on the camera for a couple years. But that's the type of school he does go to"},
    {"name": "Paper", "example": " [14] L R Alexander, et al. Experiments with genetically modified Arabidopsis showing impacts of active oxygen on abiotic stress tolerance. Plant biology 2014,16, 931-942.\n[15] S Wu, B Chauhan, D U Mamet, B Ogunwuola, S S Israely, E G Daniel, K S Nyakai. Enhanced resilience of young rice plants against drought by judicious"},
    {"name": "Wiki", "example": "# Lewis Hamilton (cricket)\n\nLewis Ashburner Hamilton-Killearn (born 25 July 2001) is a Northern Ireland international cricketer who plays for Sussex in List A and Twenty20 cricket.\n\nFor the Formula One racing driver, see Lewis Hamilton.\n\nLewis Ashburner Hamilton-Killearn\n\nHamilton started playing cricket at youth level in the Cornwall Under-16s, and later progressed through the Cornwall Youth Cricket League. He began his senior career in Wales, playing for a"},
    {"name": "Multilingual (Not English/Chinese)", "example": "المزيد\nالصحة 01 06 2016 | اليفر نيوز\nالاثنين 11 يونيو 2016 - 12:00 بض\nحلية الديوانية تسجل 69 حالات ضرورة الرشادة في خراسان\nوضع ارقام وزارة الصحة والسمرغومي أودى بتسجيل 69 حالة ضرورة الرشادة في خراسان، بينها,"},
    {"name": "Completely nonsense text", "example": "[700] [701] [702] [703] [704] [705] [706] [707] [708] [709] [710]"},
]

async def classify_sample(client, sample, semaphore):
    async with semaphore:
        try:
            messages = [
                {"role": "system", "content": "Classify the following text into exactly one of these categories:\n\n" + "\n\n".join([f"Category: {t['name']}\nExample: {t['example']}" for t in TAXONOMY]) + "\n\nReturn only the category name."},
                {"role": "user", "content": sample}
            ]
            response = await client.chat.completions.create(
                model="gpt-5",
                messages=messages
            )
            category = response.choices[0].message.content.strip()
            # Basic validation to ensure the category is in the taxonomy
            if category not in [t['name'] for t in TAXONOMY]:
                raise ValueError(f"Invalid category: {category}")
            return category
        except Exception as e:
            print(f"Error classifying sample: {e}")
            return "Error"

async def process_file(client, filepath, semaphore):
    print(f"Processing {filepath}...")
    try:
        with open(filepath, 'r') as f:
            samples = json.load(f)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None

    tasks = [classify_sample(client, sample, semaphore) for sample in samples]
    categories = await asyncio.gather(*tasks)

    categorized_data = []
    for sample, category in zip(samples, categories):
        categorized_data.append({"sample": sample, "category": category})

    # Save categorized output
    base_name = os.path.basename(filepath).replace('.json', '')
    output_path = os.path.join(os.path.dirname(filepath), f"{base_name}_categorized.json")
    
    with open(output_path, 'w') as f:
        json.dump(categorized_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {output_path}")
    return base_name, categories

import matplotlib.pyplot as plt

def plot_stats(csv_path, output_path):
    # Read CSV
    stats = defaultdict(lambda: defaultdict(int))
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats[row['model']][row['category']] = int(row['num_of_category'])

    models = ['gpt2-large', 'llama-3.2-1b', 'qwen3-0.6b']
    
    # Category mapping
    category_map = {
        "Completely nonsense text": "Gibberish",
        "Multilingual (Not English/Chinese)": "Multilingual",
        "Advertisement": "Ad"
    }

    # Define consistent colors
    # Using tab10 colormap which has 10 distinct colors
    all_categories = [
        "News", "QA", "Code", "Ad", "Social Media", 
        "Paper", "Wiki", "Multilingual", "Gibberish"
    ]
    cmap = plt.get_cmap('tab10')
    color_map = {cat: cmap(i) for i, cat in enumerate(all_categories)}

    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for ax, model in zip(axes, models):
        model_stats = stats.get(model, {})
        labels = []
        sizes = []
        for cat, count in model_stats.items():
            if count > 0:
                # Apply mapping or use original name
                display_name = category_map.get(cat, cat)
                labels.append(display_name)
                sizes.append(count)
        
        if not sizes:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=20)
            ax.set_title(model, fontsize=24)
            continue

        # Get colors for the current labels
        current_colors = [color_map.get(label, '#dddddd') for label in labels]

        ax.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=140, 
               textprops={'fontsize': 20}, colors=current_colors)
        ax.set_title(model, fontsize=24)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Pie charts saved to {output_path}")

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = AsyncOpenAI(api_key=api_key)
    
    # Find all sample files, excluding already categorized ones
    files = glob.glob("samples/*.json")
    target_files = [f for f in files if not f.endswith("_categorized.json") and "taxonomy_stats" not in f]
    
    if not target_files:
        print("No sample files found to process.")
        # Even if no new files, we might want to regenerate stats/plots if requested, 
        # but for now let's assume we proceed to stats generation if we have data.
    
    semaphore = asyncio.Semaphore(10) # Limit concurrent requests
    
    results = []
    for filepath in target_files:
        base_name = os.path.basename(filepath).replace('.json', '')
        output_path = os.path.join(os.path.dirname(filepath), f"{base_name}_categorized.json")
        if os.path.exists(output_path):
            print(f"Skipping {filepath} as {output_path} already exists.")
            # We still need to load the categories for stats
            with open(output_path, 'r') as f:
                data = json.load(f)
                categories = [item['category'] for item in data]
                # Create a dummy future that returns the result
                f = asyncio.Future()
                f.set_result((base_name, categories))
                results.append(f)
        else:
            results.append(process_file(client, filepath, semaphore))
    
    if results:
        processed_results = await asyncio.gather(*results)
        
        # Aggregate stats
        stats = defaultdict(lambda: defaultdict(int))
        model_total_counts = defaultdict(int)

        for res in processed_results:
            if res is None:
                continue
            model_name, categories = res
            model_total_counts[model_name] = len(categories)
            for cat in categories:
                stats[model_name][cat] += 1

        # Write stats to CSV
        csv_path = "samples/taxonomy_stats.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['model', 'category', 'num_of_category'])
            
            for model in stats:
                for item in TAXONOMY:
                    category = item['name']
                    count = stats[model].get(category, 0)
                    writer.writerow([model, category, count])

        print(f"Stats saved to {csv_path}")
        
        # Generate Plot
        plot_stats(csv_path, "samples/taxonomy_stats.png")
    else:
        # If no results (e.g. no files found initially, though the check above handles it),
        # check if we can just plot existing stats.
        csv_path = "samples/taxonomy_stats.csv"
        if os.path.exists(csv_path):
             plot_stats(csv_path, "samples/taxonomy_stats.png")

if __name__ == "__main__":
    asyncio.run(main())
