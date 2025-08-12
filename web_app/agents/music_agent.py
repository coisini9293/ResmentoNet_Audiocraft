"""
音乐推荐智能体
使用LangChain框架实现
"""

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from typing import List, Union, Dict, Any
import re
import json
import time

class MusicRecommendationAgent:
    """音乐推荐智能体"""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """
        初始化音乐推荐智能体
        
        Args:
            llm_model: 使用的LLM模型
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        self.tools = self._create_tools()
        self.prompt = MusicRecommendationPromptTemplate(
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )
        
        # 创建智能体
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=MusicRecommendationOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """创建工具列表"""
        return [
            Tool(
                name="music_prompt_generator",
                func=self._generate_music_prompt,
                description="根据情绪和目标生成音乐生成提示词"
            ),
            Tool(
                name="music_style_analyzer",
                func=self._analyze_music_style,
                description="分析音乐风格和特征"
            ),
            Tool(
                name="emotion_music_matcher",
                func=self._match_emotion_to_music,
                description="匹配情绪和音乐类型"
            ),
            Tool(
                name="music_effect_predictor",
                func=self._predict_music_effect,
                description="预测音乐对情绪的影响"
            )
        ]
    
    def _generate_music_prompt(self, emotion_data: str) -> str:
        """
        生成音乐提示词
        
        Args:
            emotion_data: 情绪数据（JSON字符串）
            
        Returns:
            str: 音乐生成提示词
        """
        try:
            data = json.loads(emotion_data)
            emotion = data.get('emotion', '')
            target_emotion = data.get('target_emotion', '')
            
            prompt_templates = {
                'sad': {
                    'happy': [
                        "An upbeat, joyful instrumental with energetic piano and cheerful melody, perfect for lifting spirits",
                        "A happy, uplifting song with bright acoustic guitar and positive rhythm, designed to bring joy",
                        "An optimistic, energetic tune with lively percussion and warm harmonies, ideal for mood improvement"
                    ]
                },
                'angry': {
                    'peaceful': [
                        "A peaceful, calming instrumental with soft piano and gentle strings, designed to soothe anger",
                        "A tranquil, meditative piece with slow tempo and soothing harmonies, perfect for relaxation",
                        "A relaxing melody with gentle acoustic guitar and peaceful atmosphere, ideal for stress relief"
                    ]
                },
                'fear': {
                    'calm': [
                        "A gentle, calming instrumental piece with soft piano and strings, peaceful and soothing",
                        "A tranquil ambient music with nature sounds, very relaxing and peaceful",
                        "A slow, meditative melody with gentle wind chimes and soft harmonies, designed for comfort"
                    ]
                },
                'disgust': {
                    'pleasant': [
                        "A cheerful, uplifting instrumental with bright acoustic guitar and light percussion",
                        "A happy, positive melody with gentle piano and warm harmonies, refreshing and pleasant",
                        "An optimistic tune with soft strings and gentle rhythm, designed to improve mood"
                    ]
                }
            }
            
            if emotion in prompt_templates and target_emotion in prompt_templates[emotion]:
                import random
                prompt = random.choice(prompt_templates[emotion][target_emotion])
                return f"推荐提示词：{prompt}"
            else:
                return "使用通用舒缓音乐提示词：A gentle, calming instrumental piece with soft piano and strings, peaceful and soothing"
                
        except Exception as e:
            return f"生成提示词失败：{str(e)}"
    
    def _analyze_music_style(self, music_info: str) -> str:
        """
        分析音乐风格
        
        Args:
            music_info: 音乐信息
            
        Returns:
            str: 风格分析结果
        """
        try:
            data = json.loads(music_info)
            emotion = data.get('emotion', '')
            target_emotion = data.get('target_emotion', '')
            
            style_analysis = f"""
音乐风格分析：
- 当前情绪：{emotion}
- 目标情绪：{target_emotion}

推荐音乐特征：
"""
            
            if target_emotion == 'happy':
                style_analysis += """
- 节奏：快节奏（120-140 BPM）
- 调性：大调，明亮色彩
- 乐器：钢琴、吉他、打击乐
- 情感：积极、欢快、充满活力
- 用途：提升心情、增加能量
"""
            elif target_emotion == 'peaceful':
                style_analysis += """
- 节奏：慢节奏（60-80 BPM）
- 调性：小调，柔和色彩
- 乐器：钢琴、弦乐、自然音效
- 情感：平静、和谐、放松
- 用途：缓解压力、促进睡眠
"""
            elif target_emotion == 'calm':
                style_analysis += """
- 节奏：中慢节奏（70-90 BPM）
- 调性：自然调式，温暖色彩
- 乐器：钢琴、长笛、环境音效
- 情感：安全、温暖、舒适
- 用途：缓解焦虑、提供安全感
"""
            elif target_emotion == 'pleasant':
                style_analysis += """
- 节奏：中等节奏（90-110 BPM）
- 调性：大调，清新色彩
- 乐器：吉他、钢琴、轻打击乐
- 情感：愉悦、清新、轻松
- 用途：改善心情、创造愉悦感
"""
            
            return style_analysis
            
        except Exception as e:
            return f"音乐风格分析失败：{str(e)}"
    
    def _match_emotion_to_music(self, emotion: str) -> str:
        """
        匹配情绪和音乐类型
        
        Args:
            emotion: 情绪类型
            
        Returns:
            str: 匹配结果
        """
        matching = {
            'sad': {
                'primary': 'uplifting_pop',
                'secondary': 'classical_allegro',
                'description': '欢快流行音乐或古典快板，帮助提升心情',
                'duration': '3-5分钟',
                'intensity': '中等强度'
            },
            'angry': {
                'primary': 'ambient_meditation',
                'secondary': 'nature_sounds',
                'description': '环境冥想音乐或自然音效，帮助平复情绪',
                'duration': '5-10分钟',
                'intensity': '低强度'
            },
            'fear': {
                'primary': 'soft_piano',
                'secondary': 'warm_strings',
                'description': '柔和钢琴曲或温暖弦乐，提供安全感',
                'duration': '4-6分钟',
                'intensity': '低强度'
            },
            'disgust': {
                'primary': 'acoustic_folk',
                'secondary': 'light_instrumental',
                'description': '民谣音乐或轻音乐，清新愉悦',
                'duration': '3-4分钟',
                'intensity': '中等强度'
            }
        }
        
        if emotion in matching:
            info = matching[emotion]
            return f"""
情绪-音乐匹配：
- 情绪：{emotion}
- 主要音乐类型：{info['primary']}
- 备选音乐类型：{info['secondary']}
- 描述：{info['description']}
- 推荐时长：{info['duration']}
- 推荐强度：{info['intensity']}
"""
        else:
            return f"未知情绪：{emotion}，建议使用通用舒缓音乐"
    
    def _predict_music_effect(self, music_data: str) -> str:
        """
        预测音乐对情绪的影响
        
        Args:
            music_data: 音乐数据
            
        Returns:
            str: 影响预测
        """
        try:
            data = json.loads(music_data)
            emotion = data.get('emotion', '')
            target_emotion = data.get('target_emotion', '')
            
            effects = {
                'sad_to_happy': {
                    'immediate': '提升能量水平，改善心情',
                    'short_term': '增加积极思维，促进社交欲望',
                    'long_term': '建立积极情绪模式，增强心理韧性',
                    'mechanism': '通过快节奏和明亮调性激活大脑奖励系统'
                },
                'angry_to_peaceful': {
                    'immediate': '降低心率，缓解肌肉紧张',
                    'short_term': '促进理性思考，改善决策能力',
                    'long_term': '培养情绪调节能力，减少冲动行为',
                    'mechanism': '通过慢节奏和和谐旋律激活副交感神经系统'
                },
                'fear_to_calm': {
                    'immediate': '提供安全感，缓解焦虑',
                    'short_term': '改善注意力集中，增强自信心',
                    'long_term': '建立安全感，减少恐惧反应',
                    'mechanism': '通过温暖音色和稳定节奏激活安全系统'
                },
                'disgust_to_pleasant': {
                    'immediate': '转移注意力，改善心情',
                    'short_term': '促进积极体验，增强愉悦感',
                    'long_term': '培养积极情绪，改善整体幸福感',
                    'mechanism': '通过清新旋律和愉悦音色激活愉悦系统'
                }
            }
            
            key = f"{emotion}_to_{target_emotion}"
            if key in effects:
                effect = effects[key]
                return f"""
音乐效果预测：
- 情绪转换：{emotion} → {target_emotion}

即时效果：
{effect['immediate']}

短期效果：
{effect['short_term']}

长期效果：
{effect['long_term']}

作用机制：
{effect['mechanism']}
"""
            else:
                return f"情绪转换：{emotion} → {target_emotion}，预期产生积极影响"
                
        except Exception as e:
            return f"效果预测失败：{str(e)}"
    
    def recommend_music(self, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        推荐音乐
        
        Args:
            emotion_data: 情绪数据
            
        Returns:
            Dict[str, Any]: 推荐结果
        """
        try:
            # 准备输入数据
            input_data = {
                "emotion": emotion_data.get('emotion', ''),
                "target_emotion": emotion_data.get('target_emotion', ''),
                "confidence": emotion_data.get('confidence', 0),
                "timestamp": time.time()
            }
            
            # 构建查询
            query = f"""
请为以下情绪状态推荐合适的音乐：
当前情绪：{input_data['emotion']}
目标情绪：{input_data['target_emotion']}
置信度：{input_data['confidence']}

请使用以下工具进行分析：
1. 使用music_prompt_generator生成音乐提示词
2. 使用music_style_analyzer分析音乐风格
3. 使用emotion_music_matcher匹配情绪和音乐
4. 使用music_effect_predictor预测音乐效果
"""
            
            # 执行智能体
            result = self.agent_executor.run(query)
            
            return {
                'success': True,
                'recommendation': result,
                'emotion_data': input_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'emotion_data': emotion_data
            }

class MusicRecommendationPromptTemplate(StringPromptTemplate):
    """音乐推荐提示模板"""
    
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.get("intermediate_steps")
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
        
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        return f"""
你是一个专业的音乐推荐智能体，专门根据情绪状态推荐合适的音乐。

可用工具：
{tools_str}

历史操作：
{thoughts}

当前输入：{kwargs["input"]}

请根据输入选择合适的工具进行音乐推荐。格式如下：

Action: 工具名称
Action Input: 工具输入

或者如果推荐完成，使用：
Final Answer: 最终推荐结果

Action:"""

class MusicRecommendationOutputParser:
    """音乐推荐输出解析器"""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """解析智能体输出"""
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        
        if not match:
            return AgentFinish(
                return_values={"output": "无法解析操作"},
                log=text,
            )
        
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text) 