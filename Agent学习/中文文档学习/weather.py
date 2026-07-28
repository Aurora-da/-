import requests

class WeatherAgent:
    def __init__(self):
        self.memory = []
        self.tools = {
            'get_weather': self.get_weather_api,
            'give_advice':self.generate_advice
        }

    # 工具1：调用天气API
    def get_weather_api(self, city):
        """调用外部的天气API获取数据"""
        # 这里模拟一个API调用
        print(f"[Agent 行动] 正在查询{city}的天气...")
        # 假设返回的数据
        mock_data = {'city': city, 'temp':22, 'condition':'晴朗', 'wind':"3级"}
        return mock_data

    # 工具2：根据天气生成建议
    def generate_advice(self, weather_data):
        """根据天气数据生成穿衣建议"""
        temp = weather_data['temp']
        condition = weather_data['condition']
        advice = f"当前{weather_data['city']}气温{temp}℃，天气{condition}。"
        if temp > 25:
            advice += "建议穿短袖、短裤。"
        elif temp > 15:
            advice += "建议穿长袖T恤、薄外套。"
        else:
            advice += "建议穿毛衣、厚外套。"
        return advice

    # 规划与执行核心
    def run(self, user_input):
        """解析用户目标并执行任务"""
        print(f"[用户指令]{user_input}")

        # 步骤1：规划 - 从命令中提取关键信息（城市）
        if "天气" in user_input and "北京" in user_input:
            city = "北京"
        else:
            return "请告诉我你想查询哪个城市的天气？"

        # 步骤2：行动 - 调用工具获取天气
        weather_info = self.tools['get_weather'](city)
        self.memory.append({'step':'fetched_weather', 'data':weather_info})

        # 步骤3：行动 - 调用工具生成建议
        final_advice = self.tools['give_advice'](weather_info)
        self.memory.append({'step':'generate_advice', 'data':final_advice})

        return final_advice

agent = WeatherAgent()
result = agent.run('我想知道北京的天气，该怎么穿衣服？')
print(f'[Agent 回复] {result}')