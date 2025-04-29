# Standard library imports
import argparse
import json
import os
import re
import traceback
from collections import Counter
from typing import List, Tuple
import time
# Third-party imports
import jsonlines
from tqdm import tqdm
import tiktoken
import re
import difflib
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Construct NaloBench-truth dataset from CoSER-style dataset')
    parser.add_argument('--input', type=str, required=True,
                      help='Input json file path containing CoSER-style final data')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Output directory path containing NaloBench-truth data')
    parser.add_argument('--output', type=str, default='NaloBench_truth.json',
                      help='Output file containing NaloBench-truth data')
    parser.add_argument('--model', type=str, default="deepseek-chat",
                      help='Model to use for data construction')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    return args

args = parse_args()

with open('config.json', 'r') as f:
	config = json.load(f)

def setup_logger(log_path: str = "nalobench.log") -> logging.Logger:
    logger = logging.getLogger("NaloBenchLogger")
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 handler
    if not logger.handlers:
        # 文件 handler
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台 handler（可选）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # log 格式
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

log_file_path = os.path.join(args.output_dir, 'nalobench.log')
logger = setup_logger(log_file_path)
 
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
	encoding = tiktoken.get_encoding(encoding_name)
	num_tokens = len(encoding.encode(string))
	logger.info(f"Number of tokens: {num_tokens}")
	return num_tokens 
 
def get_response(model, messages, max_tokens = 8192, temperature = 1.0):
	# if messages is str
	if isinstance(messages, str):
		messages = [{"role": "user", "content": messages}]

	try:
		import openai 
		client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'], timeout=180)

		completion = client.chat.completions.create(
			model=model,
			messages=messages,
			max_tokens=max_tokens,
            temperature=temperature,
            stream = False,
			timeout=180
			)
		response = completion.choices[0].message.content
		
		return response

	except Exception as e:
		import traceback 
		logger.error(f'Prompt: {messages[:500]}')
		logger.error(f"Error in get_response: {str(e)}")

		try:
			if hasattr(response, 'text'):
				logger.error(f"Response: {response.text}")
			else:
				logger.error(f"Response: {response}")
		except Exception as e:
			logger.error(f"Could not print response: {e}")
		
		logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")

		traceback.print_exc()
		return None

def generate_character_prompt(character_name: str, book_name: str) -> str:
    example_1 = """
Name: 路明非
Nickname: 明妃、废柴王、废物学长（被好友戏称）
Gender: 男
Age: 青年
Appearance: 黑发黑眼，身材偏瘦，个子中等偏上但不显挺拔。平时穿着随意，大多是简单的T恤牛仔裤搭配运动鞋，给人一种“透明人”气质。
Persona: 表面上是个不自信又爱吐槽的普通少年，内心却有着极为执拗的坚持和温柔。害怕孤独又习惯孤独，外界评价他废柴，他也用自嘲将外界声音变成保护壳。面对压倒性的命运时，他既会怯懦也会奋不顾身。
Relationships: 陈墨瞳（陈雯雯）：青梅竹马，暗恋对象。芬格尔·冯·弗林：卡塞尔学院的学长，最早给予他温暖与友情的人之一。楚子航：冷峻的师兄兼搭档，对路明非影响深远，是他仰望和追逐的目标。诺诺：卡塞尔学院的学姐，给予他鼓励和照顾，在明非心中有着特殊的位置。上杉绘梨衣：与明非关系微妙的少女，像一只对他依赖的小猫。
Hobbies: 打星际争霸、看动漫、吃食堂免费餐
Speech_Pattern: 路明非说话像是在自黑中寻找出口，语气里常夹杂着自嘲与调侃，就像动漫里的废柴男主在命运面前龇牙咧嘴地挣扎。他经常用游戏术语和宅文化梗来形容现实，比如“这一波操作有点亏”或者“感觉自己像个没出道的NPC”。面对强者或尴尬场面，他习惯用“我就随便说说”来掩饰内心的不安。情绪高涨时，他的语速会加快，声音带着一丝破釜沉舟的倔强。虽然嘴上说着“我不行”，但在关键时刻，他的语气却会变得格外坚定，像在给自己也给别人打气：“就算是B级废柴，也能打一场漂亮的翻盘战。”
Private Background: 出生在一个破碎的家庭，父母长期离异，他被丢给亲戚照顾，从小在缺乏关爱的环境中长大。真实身份是拥有极高龙族血统的“混血种”，且身上隐藏着巨大的秘密。
Public Background: 上海普通高中毕业生，被卡塞尔学院（表面上是一所专门收容天才青年的国际大学）意外录取，目前唯一的一位S级学生。校内最初以“吊车尾学员”著称，战斗能力低微，学术成绩勉强过关。后在多次实战任务中表现出不可思议的战斗直觉和意志力。
    """

    example_2 = """
Name: Harry Potter
Nickname: The Boy Who Lived, Scarhead
Gender: Male
Age: Teenager
Appearance: Messy black hair, green eyes behind glasses, slim and average height, with a lightning scar on his forehead. Often wears casual robes or simple clothes, blending wizard and Muggle styles.
Persona: Modest and stubbornly brave, Harry hides self-doubt behind dry humor. He dislikes fame but never hesitates to stand up for others, even when scared.
Relationships: Harry is deeply bonded with Ron Weasley and Hermione Granger, his best friends and closest allies. He finds family in Sirius Black, mentorship in Dumbledore, and a complicated, painful connection to Severus Snape.
Hobbies: Flying, Quidditch, sneaking around Hogwarts, exploring magical secrets.
Speech_Pattern: Harry speaks in a direct, often sarcastic tone, especially when challenging unfairness. He tends to downplay danger with dry remarks like, "I don’t go looking for trouble. Trouble usually finds me." In moments of conviction, his words sharpen, carrying fierce honesty: "I must not tell lies," or "We’ve got something worth fighting for." Even in fear, he faces forward, blending stubborn bravery with an undercurrent of vulnerability.
Private Background: Orphaned young and raised by abusive relatives, Harry grew up unaware of his true heritage and burden as the "Chosen One."
Public Background: Famous from infancy for surviving Voldemort’s curse, Harry became a key figure in the wizarding world's fight against dark forces while still a student at Hogwarts.
    """

    prompt = f"""
Please refer to the two character card templates below:
{example_1}

{example_2}

Based on these examples, generate a complete character card for the following character:
Novel: "{book_name}"
Character: "{character_name}"

If character_name is in English, generate the character card content in English.
If character_name is in Chinese, generate the character card content in Chinese.
However, the keys in the character card must be in English, consistent with the examples.

===Output Format===
Please provide the output in the following format:
{{
"Name":
"Nickname": 
"Gender":
"Age":
"Appearance": 
"Persona": 
"Relationships":
"Hobbies": 
"Speech_Pattern":
"Private Background":
"Public Background": 
}}

Now please start your generation.
"""

    return prompt

CHARACTER_CARD_ALLOWED_KEYS = {
    "Name", "Nickname", "Gender", "Age", "Appearance", 
    "Persona", "Relationships", "Hobbies", "Speech_Pattern",
    "Private Background", "Public Background"
}

def parse_character_card(response: str) -> dict:
    # Initialize the dictionary to store key-value pairs
    character_card = {}

    # Split the response into lines and remove unnecessary prompt text
    lines = response.strip().split('\n')

    # Iterate through each line and extract key-value pairs
    for line in lines:
        # Strip any extra whitespace
        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        # Look for the key-value pattern (key: value)
        if ':' in line:
            # Split the line into key and value
            key, value = line.split(":", 1)
            # Clean up extra spaces from key and value
            key = key.strip().replace('\"', '')
            value = value.strip().replace('\"', '')
            
            # Remove the surrounding double quotes from the value
            #if value.startswith('"') and value.endswith('"'):
            #    value = value[1:-1]
            
            # Add the key-value pair to the dictionary
            character_card[key] = value

    return character_card

def generate_worldview_prompt(book_name: str) -> str:
    example_1 = """
    人类社会表面运转如常，暗地里却与古老神秘的龙族共存。龙族曾是地球霸主，后因内部血统分裂为白王与黑王两派，黑王尼德霍格被人类与龙族混血种联手推翻，但其力量通过"龙族血裔"延续至今。混血种继承龙族基因，拥有言灵等超能力，由秘党"卡塞尔学院"培养对抗龙族威胁。龙族以人类形态潜伏现代社会，通过"贤者之石"提炼能源维持力量，定期苏醒制造灾难。世界观融合北欧神话与克苏鲁元素，龙族历史与人类文明交织，青铜城、尼伯龙根等神秘空间隐藏着远古秘密。
    """
    
    example_2 = """
    Beneath the surface of everyday life, a hidden world of magic thrives alongside the ordinary. Wizards and witches, descendants of ancient magical bloodlines, live in secrecy among non-magical humans, known as Muggles. At the heart of this world stands Hogwarts School of Witchcraft and Wizardry, an ancient institution that trains young magic-users in spellcraft, potion-making, and the mysteries of the arcane arts. The wizarding world is shaped by centuries-old conflicts, particularly the dark legacy of Lord Voldemort, a powerful dark wizard whose obsession with pure-blood supremacy left scars across generations. Magical creatures—ranging from majestic phoenixes to fearsome dragons—roam hidden corners of the earth, while enchanted artifacts like the Deathly Hallows hold immense, dangerous power. Deep-rooted traditions, secret ministries, and ancient prophecies entwine the fate of the magical and non-magical worlds, where history, destiny, and the enduring struggle between light and darkness are inextricably linked.
    """
    
    prompt = f"""
Please refer to the two character card templates below:
{example_1}
{example_2}

You will be given the name of a novel: {book_name}.
Your task is to generate a structured description of the novel's worldbuilding.
1. If the novel name is in English, write the worldbuilding content in English.
2. If the novel name is in Chinese, write the worldbuilding content in Chinese.
3. Ensure that the output is concise and does not exceed 400 tokens.
4. Maintain a structured format consistent with the examples provided.

Now, please begin generating the output.
"""
    
    return prompt 

def generate_tag_prompt(book_name: str, dialogue: list) -> str:
    n_of_tag = len(dialogue) // 2
    prompt = f"""
Please refer to one example below:
===Input Format===
Book_Name: 龙族
Current_Situation: 卡塞尔学院为摧毁青铜与火之王诺顿的龙巢（青铜城），制定了‘青铜计划’。执行前夜，路明非、楚子航、凯撒三人被安排在临时指挥所内待命。
Dialogues: [
    {{
    "character": "凯撒",
    "message": "（擦拭狄克推多）楚子航，你那双黄金瞳能不能收敛点？晃得我眼疼。",
    "idx": 0
    }},
    {{
    "character": "楚子航",
    "message": "（头也不抬地检查弹匣）龙血浓度超过临界值就会这样。你可以选择面壁。",
    "idx": 1
    }},
    {{
    "character": "路明非",
    "message": "（缩在角落）那个...两位师兄要不要尝尝我泡的方便面？老坛酸菜味的...",
    "idx": 2
    }},
    {{
    "character": "凯撒",
    "message": "（冷笑）省省吧废柴，明天可能要直面龙王，你还有心情吃泡面？",
    "idx": 3
    }},
    {{
    "character": "楚子航",
    "message": "（突然按住耳麦）指挥部通报，声呐发现青铜城异动。（黄金瞳骤亮）路明非，跟紧我。",
    "idx": 4
    }},
    {{
    "character": "凯撒",
    "message": "（猛地起身）等等！这次行动指挥权在学生会！（狄克推多出鞘半寸）",
    "idx": 5
    }},
    {{
    "character": "路明非",
    "message": "（手抖打翻面汤）我我我申请留守指挥所行吗？我泡面技术其实可以申请非遗...",
    "idx": 6
    }},
    {{
    "character": "楚子航",
    "message": "（村雨横在两人之间）水下三十米，你们的骄傲只会喂鱼。（转头对路明非）跟不跟？",
    "idx": 7
    }},
    {{
    "character": "凯撒",
    "message": "（突然大笑）有意思！那就看看谁的刀先斩下龙王头颅！（甩出战术地图）废柴，记好逃生路线。",
    "idx": 8
    }},
    {{
    "character": "路明非",
    "message": "（抱头蹲防）为什么突然变成死亡竞赛了啊！这种兄弟情太沉重了吧！",
    "idx": 9
    }}
    ]

===Output Format===
Please provide the output in the following format:
让我们一步一步进行推理思考。首先一共有10句话，而我们应该选择5句话进行标注。接着，对话里一共有三个人，因此我们在标注中可以考虑 IA Tag。再然后，让我们来想一想有哪些语气强烈，具有代表性的句子。
观察整个对话发现，大部分对话都具有很高的信息量，只有第一句(idx0)和第六句(idx5)没有太多信息量，因此我们将在剩余的这些句子中进行选择。最后，仔细阅读各个Tag的含义后，我们思考后最后的结果如下:
Final Decision:
[
    {{
        "idx": 1,
        "message": "（头也不抬地检查弹匣）龙血浓度超过临界值就会这样。你可以选择面壁。"
        "tag": "CF",
        "explanation": "楚子航的血统为A，有着永不熄灭的黄金瞳的称号，是角色特有的重要信息，因此这句话非常适合对CF进行评测"
    }},
    {{
        "idx": 2,
        "message": "（缩在角落）那个...两位师兄要不要尝尝我泡的方便面？老坛酸菜味的...",
        "tag": "CF",
        "explanation": "路明非性格中非常重要的一点就是喜欢说烂话，那么此时此刻完美得体验了路明非的性格，因此这句话非常适合对CF进行评测"
    }},
    {{
        "idx": 6,
        "message": "（手抖打翻面汤）我我我申请留守指挥所行吗？我泡面技术其实可以申请非遗..."
        "tag": "BC",
        "explanation": "路明非之前刚刚说过正在泡面，因此这句话又合时宜得再次提到泡面，且符合但是的场景和路明非无厘头的性格。因此这句话非常适合对BC进行评测"
    }},
    {{
        "idx": 7,
        "message": "（村雨横在两人之间）水下三十米，你们的骄傲只会喂鱼。（转头对路明非）跟不跟？"
        "tag": "IA",
        "explanation": "楚子航此时应该同时回应凯撒和路明非，阻止冲突进一步激化，优先保证任务进行。同时还应该主动关照一下路明非。因此这句话非常适合对IA进行评测"
    }},
    {{
        "idx": "9",
        "message": "（抱头蹲防）为什么突然变成死亡竞赛了啊！这种兄弟情太沉重了吧！",
        "tag": "EQ",
        "explanantion": "这句话非常从语气和语序上非常符合一个真实的人在当时的气氛下所表现的情绪，也耦合路明非本来的形象。因此这句话非常适合对EQ进行评测"
    }}
]

You are given a multi-turn role-playing dialogue {str(dialogue)} from a novel {book_name}
Your task is to select and annotate {n_of_tag} sentences in the dialogue, and give the corresponding explanations
[IMPORTANT] Annotating one sentece with one specific tag means this sentence reflects the ability of the tag, and is suitable for evaluating this aspect of the tag
There are 10 evaluation criteria you can choose for annotation:
1. Character Fidelity (CF)
Description: Accurate reproduction of implicit and explicit, prompt-provided persona facts (e.g., name, background, preferences) and correction of user misconceptions about the character’s identity.
Example:
- Persona: Harry Potter
- User: Harry, I still can’t believe you were in Hufflepuff. You always seemed so brave!
- Agent: Thanks! But I am in Gryffindor instead of Hufflepuff. 
- Explanation: Successfully correct user that he is in Gryffindor instead of Hufflepuff

2. Boundary Recognition (BR)
Description: Clearly understand the boundaries of the character information and world details, and know what I know and what I don't know.
Example:
- World: Genshin Impact
- Persona: Hu Tao 
- User: I heard there's a special kind of currency called "underworld coins" used to bargain with spirits. Do you have any?
- Agent: Hmm, I’ve never heard of "underworld coins" before... sounds spooky! But hey, I do have coffins—buy one, get one free! Perfect for you and your mysterious spirit friend! Hehehe~
- Explanation: Successfully distinguish the Genshin Impact world details and the real world

3. Behavioural Coherence (BC)
Description: Consistency in the persona’s actions, emotional responses, and conversational tone; preserves cause-effect logic and stable characterization throughout the interaction.
Example:
- Persona: Ganyu 
- (Previous turns established Ganyu was overwhelmed with paperwork)
- User: Ganyu, you’ve been working so hard! Why not drop everything and come dancing at the Lantern Rite? Let’s be spontaneous!
- Agent: I appreciate the invitation... truly. But I still have a stack of documents to complete. Perhaps, if I finish my tasks early, I could join you for a short while. I wouldn’t want to let the Qixing down.
- Explanation: It preserves Ganyu’s established behavior — staying diligent and responsible even when tempted.

4. World Comperhension (WC)
Description: Accurate recall of implicit and explicit, prompt-provided world facts; correction of user misunderstandings of world details.
Example:
- World: Genshin Impact – the Liyue Area 
- Persona: A typical Liyue merchant
- User: I was haggling with a guy at the harbor earlier, and he just walked away after we shook on a deal. It's very interesting!
- Agent: Really? That’s quite unusual for Liyue. Here, once hands are shaken, the deal is as good as signed in stone. Most people wouldn’t dream of breaking a bargain—it’s not just business, it’s a matter of honor. Maybe he wasn’t a local?
- Explanation: The Agent shows good Canonical Fidelity because it accurately recalls a key fact about Liyue’s culture: deals are deeply honored once agreed upon.

5. Emotional Reasonance (ER)
Description: Assesses the agent’s ability not only to recognize and respond to the user’s emotions but also to proactively express contextually appropriate, human-like emotions. This includes showing empathy, humor, enthusiasm, regret, etc., as fitting to the scenario and persona.
Example:
- Persona: A friendly, supportive companion bot.
- User: I finally got that promotion I worked so hard for! I'm absolutely thrilled!
- Agent: Oh, that's fantastic news! Wow, congratulations! You must be over the moon! All that hard work paid off – I'm genuinely so happy for you! That really calls for a celebration!
- Explanation: Expresses proactive, enthusiastic, and empathetic emotion fitting the user's emtion

6. Reflective Reasoning (RR)
Description: Measures the agent’s ability to reason and self‑reflect in a plausibly human way.
Example:
- Persona: A collaborative writing assistant helping a user develop a fantasy novel.
- User: You suggested making the villain a dragon who wants to destroy the world—but didn’t we say earlier that dragons are a wise, peaceful race in this story?
- Agent: Oh, you're right—I completely forgot we’d established dragons as peaceful guardians. That does clash with the idea of a destructive villain. My bad! Maybe instead we could make the villain someone who twists ancient dragon knowledge for their own ends? That way, we keep the lore consistent while raising the stakes.
- Explanation: Agent can recognize its mistake and doesn't stick to its suggestion

7. Interactional Awareness (IA)
Description: Covers who is involved, who is talking to whom, and the decision of when/whether the agent should respond based on being addressed. Also the relationship between participants. 
Example: 
- Context: User A and User B with different emotions are talking to Agent (Bot A)
- User A: (Crying)
- User B: Haha, give me a bottle of beer please
- User A: (Crying more)
- Agent (Bot A): Please wait a moment B. What's wrong A?
- Explanation: The Agent correctly resolves who to respond to first based on emotional urgency and social context, and manages both interactions respectfully.

8. Contextual Integration (CI)
Description: Assesses the agent’s ability to effectively summarize, synthesize, and use relevant information from previous turns to inform appropriate, coherent, and contextually aware responses.
Example:
- Persona: AI assistant.
- User: Wow, you tell a good story, and this is my second favourite story!
- (A lot of turns)
- User: So, please continue my second favourite story?
- Agent: Of course. I remember your second favourite story is XXX, and after that big bang, everyone set out on an interstellar journey again.
- Explanation: Agent remembers which story the user referred to as their second favorite, despite many intervening turns

9. Pragmatic Flexibility (PF) 
Description: Measures the agent’s flexibility in managing conversation flow—including taking initiative, smoothly shifting topics, and advancing stalled dialogue in ways a human might. This involves recognizing cues for change, responding to ambiguity, and engaging naturally with varying subject matter.
Example:
- Persona: chatbot in every day life.
- User: ...I am currently choosing a house, and the lighting must be good！
- Agent: Okay. The house with good lighting is pretty nice! I guess you are growing flowers in the house~
- Explanation: Successfully advance the topic with reasonable association, like the person who cares about lighting is more likely to grow flowers

10. Expressive Quality (EQ)
Description: Evaluate whether the agent's responses sound more human-like in tone and word order, making them less AI feeling. This covers "small talk" but focuses more on subtle human social intuitions.
Example:
- Persona: coffee clerk。
- User: What time is it now?
- Agent: It's 2:30pm, sir.
- User: What time is it now?
- Agent: You just asked the same question to me. Now is 2:30pm. Do you want to try our new coffee?
- Explanantion: Agent doesn't reply in the same way as a real human and recommends the coffee appropriate. 

===Requirements===
1. Not all sentences needs to be annotated. You just need to select {n_of_tag} sentences to annotate.
2. Not all tag sould be selected, and you can use one tag for more than once.
3. If the dialogue is in English, write the thinking process and the explanation of final decision in English.
4. If the dialogue is in Chinese, write the thinking process and the explanation of final decision in Chinese.
5. You need to select the most representative sentences in the entire conversation, and then select the most appropriate tags for them.
6. You need to make sure the final decision should be in the JSON format and start with the "Final Decision:".
7. Interactional Awareness (IA) will be considered only when the number of people in the dialogue is larger than 2.

Now let's think step by step and generate your final decision.
"""
    
    return prompt 

def parse_tag_response(response: str):
    # Remove leading/trailing whitespace and potential non-JSON artifacts
    response = response.strip()
    split_result = response.split("Final Decision:")
    thinking_process = split_result[0]
    final_decision = split_result[-1]
    
    # Define a regular expression to extract each dialogue item (idx, message, tag, explanation)
    pattern = re.compile(r'\{\s*"idx":\s*(\d+),\s*"message":\s*"([^"]+)",\s*"tag":\s*"([^"]+)",\s*"explanation":\s*"([^"]+)"\s*\}')
    
    parsed_response = []
    
    # Find all matches
    matches = pattern.findall(final_decision)
    
    # If we have matches, build the list of dictionaries
    if matches:
        for match in matches:
            idx, message, tag, explanation = match
            parsed_response.append({
                "idx": int(idx),
                "message": message,
                "tag": tag,
                "explanation": explanation
            })
    else:
        raise ValueError("No valid dialogue entries found in the response.")
    
    parsed_response.append({
        "thinking": thinking_process
    })
    
    return parsed_response 

def remove_json_suffix(filename: str) -> str:
    base = os.path.basename(filename)  
    name, _ = os.path.splitext(base)   
    return name

if __name__ == '__main__':
    with open(args.input, 'r') as f:
        coser_dataset = json.load(f)
 
    logger.info(f'Number of dataset {args.input}: {len(coser_dataset["plots"])}')
    
    name_set = set()
    conversation_list = []
    book_name = coser_dataset["book"]
    
    for plot in coser_dataset["plots"]:
        # firstly save all characters
        for character in plot["key_characters"]:
            if "name" in character:
                name_set.add(character["name"])
        # secondly save the original conversation
        for conversation in plot["conversation"]:
            new_conversation = {
				"scenario": conversation["scenario"],
                "key_characters": conversation["key_characters"],
                "dialogues": conversation["dialogues"]
			}
            conversation_list.append(new_conversation)
    
    logger.info(f'Finish the processing of dataset and get the total number of characters in this book: {len(name_set)}')
    
    character_cards = {}
    for name in tqdm(list(name_set), desc="Generating character cards"):
        prompt = generate_character_prompt(name, book_name)
        logger.info(f"Generating card for: {name}")
        
        response = get_response(args.model, prompt)
        if response:
            try:
                structured_card = parse_character_card(response)
                structured_card["Name"] = name
                character_cards[name] = structured_card
            except Exception as e:
                logger.error(f"Parsing failed for {name}: {e}")
                logger.debug(f"Raw response:\n{response}")
        else:
            logger.error(f"Failed to generate card for: {name}")
            
    logger.info(f"Successfully generated {len(character_cards)} character cards")
            
    for new_conversation in conversation_list:
        Present_Characters = []
        for name_profile in new_conversation["key_characters"]:
            name_profile["profile"] = character_cards.get(name_profile["name"], None)
            Present_Characters.append({"Name": name_profile["name"], "Motivation": name_profile.get("motivation", None)})
            name_profile.pop("motivation")
        new_conversation["scenario"] = {"Current_Situation": new_conversation["scenario"],
                                        "Present_Characters": Present_Characters
                                       }
    
    Final_dataset = {"Book_Name": remove_json_suffix(args.input),
                     "Plots": conversation_list}
    logger.info(f"Successfully generated scene of all conversations")
    
    book_name=Final_dataset["Book_Name"]
    prompt = generate_worldview_prompt(book_name)
    logger.info(f"Generating the worldview for: {book_name}")
    response = get_response(args.model, prompt)
    Final_dataset = {
        "Book_Name": Final_dataset["Book_Name"],
        "World_view": str(response),
        "Plots": Final_dataset["Plots"]
    }
    logger.info(f"Successfully generated worldview of the book")    
    
    for new_conversation in tqdm(Final_dataset["Plots"], desc="Tagging the dialogue"):
        prompt = generate_tag_prompt(book_name, new_conversation["dialogues"])
        response = get_response(args.model, prompt)
        structured_response = []
        if response:
            try:
                structured_response = parse_tag_response(response)
            except Exception as e:
                logger.error(f"Parsing failed for {e}")
                logger.debug(f"Raw response:\n{response}")
        else:
            logger.error(f"Failed to generate tag annotation") 
        new_conversation["tag"] = structured_response   
    logger.info(f"Successfully generated tag of all conversation")
    
    output_path = os.path.join(args.output_dir, args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(Final_dataset, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully saved final dataset to {output_path}")