"""
情绪分析智能体
使用LangChain框架实现
"""

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from typing import List, Union, Dict, Any
import re
import json

class EmotionAnalysisAgent:
    """情绪分析智能体"""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """
        初始化情绪分析智能体
        
        Args:
            llm_model: 使用的LLM模型
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.tools = self._create_tools()
        self.prompt = EmotionAnalysisPromptTemplate(
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )
        
        # 创建智能体
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=EmotionAnalysisOutputParser(),
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
                name="emotion_analyzer",
                func=self._analyze_emotion,
                description="分析检测到的情绪，提供详细的情感和心理分析"
            ),
            Tool(
                name="emotion_mapper",
                func=self._map_emotion_to_music,
                description="将检测到的情绪映射到适合的音乐类型"
            ),
            Tool(
                name="improvement_suggestor",
                func=self._suggest_improvement,
                description="基于情绪分析提供改善建议"
            )
        ]
    
    def _analyze_emotion(self, emotion_data: str) -> str:
        """
        分析情绪数据
        
        Args:
            emotion_data: 情绪数据（JSON字符串）
            
        Returns:
            str: 分析结果
        """
        try:
            data = json.loads(emotion_data)
            emotion = data.get('emotion', '')
            confidence = data.get('confidence', 0)
            
            analysis = f"""
情绪分析结果：
- 检测到的情绪：{emotion}
- 置信度：{confidence:.2f}

详细分析：
"""
            
            if emotion == 'sad':
                analysis += """
悲伤情绪分析：
- 可能的原因：压力、失落、孤独感
- 生理表现：能量下降、食欲改变、睡眠问题
- 建议：寻求社交支持、进行轻度运动、听欢快音乐
"""
            elif emotion == 'angry':
                analysis += """
愤怒情绪分析：
- 可能的原因：挫折、不公平感、压力累积
- 生理表现：心跳加速、血压升高、肌肉紧张
- 建议：深呼吸练习、暂时离开刺激源、听平静音乐
"""
            elif emotion == 'fear':
                analysis += """
恐惧情绪分析：
- 可能的原因：不确定性、威胁感、焦虑
- 生理表现：紧张、出汗、注意力集中
- 建议：渐进式放松、正念冥想、听舒缓音乐
"""
            elif emotion == 'disgust':
                analysis += """
厌恶情绪分析：
- 可能的原因：不愉快体验、价值观冲突
- 生理表现：恶心感、回避行为
- 建议：转移注意力、环境改变、听清新音乐
"""
            
            return analysis
            
        except Exception as e:
            return f"情绪分析失败：{str(e)}"
    
    def _map_emotion_to_music(self, emotion: str) -> str:
        """
        将情绪映射到音乐类型
        
        Args:
            emotion: 情绪类型
            
        Returns:
            str: 音乐映射建议
        """
        mapping = {
            'sad': {
                'target': 'happy',
                'music_type': '欢快积极的音乐',
                'characteristics': '快节奏、明亮调性、积极旋律',
                'examples': '流行音乐、轻音乐、古典音乐中的快板'
            },
            'angry': {
                'target': 'peaceful',
                'music_type': '平静舒缓的音乐',
                'characteristics': '慢节奏、柔和调性、和谐旋律',
                'examples': '自然音效、冥想音乐、古典音乐中的慢板'
            },
            'fear': {
                'target': 'calm',
                'music_type': '舒缓平静的音乐',
                'characteristics': '稳定节奏、温暖调性、安全旋律',
                'examples': '环境音乐、轻音乐、钢琴独奏'
            },
            'disgust': {
                'target': 'pleasant',
                'music_type': '清新愉悦的音乐',
                'characteristics': '中等节奏、明亮调性、愉悦旋律',
                'examples': '民谣、轻音乐、自然音效'
            }
        }
        
        if emotion in mapping:
            info = mapping[emotion]
            return f"""
音乐映射建议：
- 当前情绪：{emotion}
- 目标情绪：{info['target']}
- 推荐音乐类型：{info['music_type']}
- 音乐特征：{info['characteristics']}
- 音乐示例：{info['examples']}
"""
        else:
            return f"未知情绪：{emotion}，建议使用通用舒缓音乐"
    
    def _suggest_improvement(self, emotion: str) -> str:
        """
        提供情绪改善建议
        
        Args:
            emotion: 情绪类型
            
        Returns:
            str: 改善建议
        """
        suggestions = {
            'sad': """
情绪改善建议：
1. 音乐疗法：听欢快、积极的音乐
2. 社交活动：与朋友家人联系
3. 运动：进行轻度有氧运动
4. 冥想：正念冥想练习
5. 创意活动：绘画、写作、手工
""",
            'angry': """
情绪改善建议：
1. 音乐疗法：听平静、舒缓的音乐
2. 深呼吸：进行深呼吸练习
3. 运动：进行剧烈运动释放能量
4. 时间管理：暂时离开刺激源
5. 沟通：冷静后与相关人员沟通
""",
            'fear': """
情绪改善建议：
1. 音乐疗法：听安全、温暖的音乐
2. 渐进式放松：逐步放松身体各部位
3. 正念练习：关注当下，减少担忧
4. 信息收集：了解恐惧源，制定应对策略
5. 寻求支持：与信任的人分享感受
""",
            'disgust': """
情绪改善建议：
1. 音乐疗法：听清新、愉悦的音乐
2. 环境改变：改变当前环境
3. 注意力转移：专注于其他活动
4. 清洁整理：整理环境，创造舒适空间
5. 积极思考：关注积极方面
"""
        }
        
        return suggestions.get(emotion, "建议进行通用放松活动")
    
    def analyze_and_suggest(self, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析情绪并提供建议
        
        Args:
            emotion_data: 情绪数据
            
        Returns:
            Dict[str, Any]: 分析结果和建议
        """
        try:
            # 准备输入数据
            input_data = {
                "emotion": emotion_data.get('emotion', ''),
                "confidence": emotion_data.get('confidence', 0),
                "timestamp": emotion_data.get('timestamp', '')
            }
            
            # 构建查询
            query = f"""
请分析以下情绪数据并提供改善建议：
情绪：{input_data['emotion']}
置信度：{input_data['confidence']}
时间：{input_data['timestamp']}

请使用以下工具进行分析：
1. 使用emotion_analyzer分析情绪
2. 使用emotion_mapper映射音乐类型
3. 使用improvement_suggestor提供改善建议
"""
            
            # 执行智能体
            result = self.agent_executor.run(query)
            
            return {
                'success': True,
                'analysis': result,
                'emotion_data': input_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'emotion_data': emotion_data
            }

class EmotionAnalysisPromptTemplate(StringPromptTemplate):
    """情绪分析提示模板"""
    
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.get("intermediate_steps")
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
        
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        return f"""
你是一个专业的情绪分析智能体，专门分析面部表情识别结果并提供改善建议。

可用工具：
{tools_str}

历史操作：
{thoughts}

当前输入：{kwargs["input"]}

请根据输入选择合适的工具进行分析。格式如下：

Action: 工具名称
Action Input: 工具输入

或者如果分析完成，使用：
Final Answer: 最终答案

Action:"""

class EmotionAnalysisOutputParser:
    """情绪分析输出解析器"""
    
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