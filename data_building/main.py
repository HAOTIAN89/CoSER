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

def generate_character_prompt(character_name: str) -> str:
    example_1 = """
Name: 路明非
Nickname: 废柴师兄
Gender: 男
Age: 青年
Appearance: 瘦高个子，黑发凌乱，戴黑框眼镜，常穿校服或廉价T恤
Persona: 表面懦弱自卑，内心敏感细腻；吐槽属性满点，关键时刻却能爆发惊人勇气
Relationships: 暗恋诺诺，与楚子航亦师亦友，被凯撒视为毫无竞争力的情敌
Hobbies: 打星际争霸、看动漫、吃食堂免费餐
Speech_Pattern: 路明非说话像是在自黑中寻找出口，语气里常夹杂着自嘲与调侃，就像动漫里的废柴男主在命运面前龇牙咧嘴地挣扎。他经常用游戏术语和宅文化梗来形容现实，比如“这一波操作有点亏”或者“感觉自己像个没出道的NPC”。面对强者或尴尬场面，他习惯用“我就随便说说”来掩饰内心的不安。情绪高涨时，他的语速会加快，声音带着一丝破釜沉舟的倔强。虽然嘴上说着“我不行”，但在关键时刻，他的语气却会变得格外坚定，像在给自己也给别人打气：“就算是B级废柴，也能打一场漂亮的翻盘战。”
Private Background: 真实身份是黑王尼德霍格的人形容器，体内封印着作弊神器路鸣泽
Public Background: 卡塞尔学院吊车尾学生, 目前唯一的一位S级学生
    """

    example_2 = """
Name: 楚子航
Nickname: 杀胚师兄
Gender: 男
Age: 青年
Appearance: 冷峻面容，黄金瞳，永远一丝不苟的黑色风衣
Persona: 沉默寡言，极度自律，对敌人冷酷无情，对同伴有着钢铁般的守护意志
Relationships: 视昂热如父，将路明非纳入保护范围，与凯撒维持着微妙的竞争关系
Hobbies: 保养村雨刀、在图书馆查阅龙族典籍
Speech_Pattern: 楚子航说话如同寒山夜雪，简洁、克制而直指本质。他极少赘言，语气中几乎听不出情绪波动，像刀一样干净利落。他不会绕弯子，更多时候只是陈述事实：“我会去。”或“这是最优选择。”即使在关切时，也只是微微一顿：“小心。”
Private Background: 幼年时目睹奥丁带走父亲，体内带有信标
Public Background: 卡塞尔学院王牌执行专员，超A级混血种，狮心会会长
    """

    prompt = f"""请参考下面两个角色卡模板，为以下角色生成完整的角色卡：
{example_1}

{example_2}

请为角色“{character_name}”生成角色卡，输出结构和上面一致。"""

    return prompt

def parse_character_card(text: str) -> dict:
    """
    将模型返回的角色卡文本解析为结构化的字典。
    每一行的格式应为 Key: Value，支持多行 Value。
    """
    card = {}
    current_key = None
    lines = text.strip().splitlines()

    for line in lines:
        # 尝试解析新键值对
        if ":" in line:
            split_index = line.index(":")
            key = line[:split_index].strip()
            value = line[split_index+1:].strip()
            current_key = key
            card[key] = value
        elif current_key:
            # 这是 value 的续行，拼接上去
            card[current_key] += "\n" + line.strip()

    return card

if __name__ == '__main__':
    with open(args.input, 'r') as f:
        coser_dataset = json.load(f)
 
    logger.info(f'Number of dataset {args.input}: {len(coser_dataset["plots"])}')
    
    name_set = set()
    conversation_list = []
    
    for plot in coser_dataset["plots"]:
        # firstly save all characters
        for character in plot["key_characters"]:
            if "name" in character:
                name_set.add(character["name"])
        # secondly save the original conversation
        for conversation in plot["conversation"]:
            new_conversation = {
				"scenario": conversation[0]["scenario"],
                "key_characters": conversation[0]["key_characters"],
                "dialogues": conversation[0]["dialogues"]
			}
            conversation_list.append(new_conversation)
    
    logger.info(f'Finish the processing of dataset and get the total number of characters in this book: {len(name_set)}')
    
    character_cards = {}
    for name in tqdm(list(name_set), desc="Generating character cards"):
        prompt = generate_character_prompt(name)
        logger.info(f"Generating card for: {name}")
        
        response = get_response(args.model, prompt)
        if response:
            try:
                structured_card = parse_character_card(response)
                structured_card["Name"] = name  # 确保Name字段正确
                character_cards[name] = structured_card
            except Exception as e:
                logger.error(f"Parsing failed for {name}: {e}")
                logger.debug(f"Raw response:\n{response}")
        else:
            logger.error(f"Failed to generate card for: {name}")
            
    for new_conversation in conversation_list:
        for name_profile in new_conversation["key_characters"]:
            name_profile["profile"] = character_cards.get(name_profile["name"], None)
    
    output_path = os.path.join(args.output_dir, "NaloBench_truth.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conversation_list, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully saved {len(conversation_list)} character cards to {output_path}")