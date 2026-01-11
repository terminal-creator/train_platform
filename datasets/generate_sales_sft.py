#!/usr/bin/env python3
"""
生成带标签的销售领域 SFT 训练数据集
支持 OpenAI messages 格式，同时添加元数据标签用于分布分析
"""

import json
import random

# 定义领域和场景
DOMAINS = {
    "旅游": {
        "system": "你是一位专业的旅游顾问，熟悉国内外旅游目的地和线路，能够为客户提供个性化的旅行规划。",
        "products": ["邮轮游", "自由行套餐", "跟团游", "定制旅行", "签证服务", "机票预订", "酒店预订"],
    },
    "健康医疗": {
        "system": "你是一位专业的健康顾问，熟悉各类医疗健康服务和产品，能够为客户提供专业的健康管理建议。",
        "products": ["健康管理", "体检套餐", "保健品", "医疗保险", "远程问诊", "康复服务"],
    },
    "B2B软件": {
        "system": "你是一位专业的B2B软件销售顾问，擅长企业软件销售，能够准确把握客户需求并提供专业的解决方案。",
        "products": ["CRM系统", "ERP系统", "OA办公", "数据分析平台", "云服务", "安全解决方案"],
    },
    "电信通讯": {
        "system": "你是一位专业的电信业务顾问，熟悉各类通信产品和套餐，能够为客户提供最优的通信解决方案。",
        "products": ["宽带服务", "5G套餐", "企业专线", "物联网卡", "云通信", "视频会议"],
    },
    "金融理财": {
        "system": "你是一位专业的金融理财顾问，熟悉各类投资理财产品，能够为客户提供稳健的财富管理方案。",
        "products": ["基金定投", "银行理财", "保险产品", "股票投资", "债券", "信托产品"],
    },
    "教育培训": {
        "system": "你是一位专业的教育顾问，熟悉各类教育培训课程，能够为学员提供最适合的学习方案。",
        "products": ["职业培训", "语言课程", "考研辅导", "IT培训", "管理课程", "兴趣班"],
    },
}

# 客户意图类型
INTENTS = {
    "产品咨询": {
        "questions": [
            "我想了解一下你们的{product}，有什么特点？",
            "能介绍一下{product}吗？",
            "{product}具体包含哪些服务？",
            "你们的{product}和市面上其他的有什么区别？",
            "{product}适合什么样的人/企业？",
        ],
        "weight": 25,
    },
    "价格询问": {
        "questions": [
            "{product}多少钱？",
            "你们的{product}价格怎么样？",
            "{product}有什么优惠活动吗？",
            "能给个{product}的报价吗？",
            "{product}性价比如何？",
        ],
        "weight": 20,
    },
    "竞品比较": {
        "questions": [
            "听说{competitor}也不错，你们和他们比怎么样？",
            "为什么我要选择你们的{product}而不是{competitor}？",
            "你们的{product}有什么独特的竞争优势？",
            "和{competitor}相比，你们的优势在哪里？",
        ],
        "weight": 15,
    },
    "使用疑虑": {
        "questions": [
            "{product}好用吗？我担心不会操作",
            "购买{product}后，有培训服务吗？",
            "{product}的售后服务怎么样？",
            "如果遇到问题怎么办？",
            "{product}有没有视频教程？",
        ],
        "weight": 15,
    },
    "效果质疑": {
        "questions": [
            "{product}真的有效果吗？",
            "能保证{product}的效果吗？",
            "有成功案例可以分享吗？",
            "你们的{product}口碑怎么样？",
            "用过的客户反馈如何？",
        ],
        "weight": 10,
    },
    "决策犹豫": {
        "questions": [
            "我再考虑考虑吧",
            "让我和家人/同事商量一下",
            "现在买合适吗？",
            "我还要对比一下其他家",
            "不着急，等等再说",
        ],
        "weight": 10,
    },
    "投诉抱怨": {
        "questions": [
            "你们的{product}体验太差了！",
            "之前买的{product}出问题了，怎么处理？",
            "服务态度不好，我要投诉",
            "说好的功能没有实现，怎么回事？",
        ],
        "weight": 5,
    },
}

COMPETITORS = ["竞品A", "竞品B", "友商", "其他品牌", "XX公司"]

# 回复模板
RESPONSE_TEMPLATES = {
    "产品咨询": """感谢您的关注！

关于{product}，我来为您详细介绍：

**核心特点：**
1. {feature1}
2. {feature2}
3. {feature3}

**适用场景：**
- {scenario1}
- {scenario2}

**客户收益：**
- {benefit1}
- {benefit2}

如果您有具体需求，我可以为您定制更详细的方案。请问您最关注哪些方面？""",

    "价格询问": """感谢您的询价！

{product}目前有以下几个版本：

**标准版：** ¥{price1}/年
- 包含基础功能
- 适合个人/小团队

**专业版：** ¥{price2}/年
- 完整功能
- 优先技术支持
- 适合中型企业

**企业版：** 定制报价
- 全部高级功能
- 专属客户经理
- 定制化开发

🎁 **本月优惠**：新客户首年享8折，还赠送{gift}！

您的预算和使用规模是怎样的？我帮您推荐最适合的方案。""",

    "竞品比较": """您考虑得非常周全，选择之前多做比较是非常明智的。

**客观分析我们的优势：**

1. **产品层面**
   - 核心技术自主研发，持续迭代优化
   - 功能覆盖更全面
   - 性能稳定，经过大量客户验证

2. **服务层面**
   - 本土化服务团队，响应更快
   - 专属客户成功经理制度
   - 完善的培训和支持体系

3. **价值层面**
   - 总体拥有成本（TCO）更优
   - 投资回报周期更短
   - 隐性成本更少

4. **案例证明**
   - 服务过{customer_count}+家客户
   - 客户续约率超过{renewal_rate}%
   - 多个行业标杆案例

当然每个品牌都有其特点，我建议您可以实际体验后再做决定。我们可以安排一次演示，您看方便吗？""",

    "使用疑虑": """感谢您的信任！

关于使用问题，请放心，我们的{product}设计理念就是"简单易用"。

**上手支持：**

1. **入门培训**
   - 购买即赠送专业培训课程
   - 视频教程+图文指南全覆盖
   - 一对一辅导快速上手

2. **日常使用**
   - 界面直观，操作简单
   - 智能引导，减少学习成本
   - 常见操作3步内完成

3. **问题解决**
   - 内置帮助中心和FAQ
   - 在线客服实时答疑
   - 远程协助快速解决

4. **持续支持**
   - 定期使用技巧分享
   - 新功能培训通知
   - 用户社群交流学习

大部分用户1-2天就能熟练使用，如果您愿意，我现在就可以给您演示一下基本操作。""",

    "效果质疑": """您的谨慎是完全可以理解的，选择产品确实需要看实际效果。

**效果保障：**

1. **数据说话**
   - 客户平均提升效率{efficiency}%
   - ROI通常在{roi_months}个月内回正
   - NPS评分达到{nps}分

2. **真实案例**
   - {case1}
   - {case2}
   - 更多案例可以查看我们的官网

3. **零风险体验**
   - 提供{trial_days}天免费试用
   - 不满意全额退款
   - 无任何隐藏费用

4. **持续优化**
   - 定期数据复盘
   - 专属成功经理跟进
   - 确保达成预期目标

要不这样，您先免费体验一下，用实际效果说话？""",

    "决策犹豫": """完全理解，这确实是一个重要的决定，需要慎重考虑。

**几点建议供您参考：**

1. **时机考量**
   - 现在正好有{promotion}活动
   - 早启动早受益，{benefit_time}
   - 市场机会不等人

2. **风险控制**
   - 支持{trial_days}天无理由退款
   - 可以先小范围试用
   - 按效果付费，无压力

3. **决策支持**
   - 我可以提供详细方案给您参考
   - 也可以安排和已有客户交流
   - 有任何疑问随时问我

您看这样好不好，我先给您发一份详细资料，您和团队商量后有任何问题随时联系我？""",

    "投诉抱怨": """非常抱歉给您带来了不好的体验，我完全理解您的心情。

**我们的处理方案：**

1. **问题解决**
   - 我已经记录了您反馈的问题
   - 会立即安排技术人员核查
   - 24小时内给您反馈处理结果

2. **补偿方案**
   - 为表歉意，赠送您{compensation}
   - 延长服务期{extend_days}天
   - 优先享受后续新功能

3. **改进承诺**
   - 您的反馈对我们很重要
   - 会推动产品/服务改进
   - 后续进展会及时同步给您

请问您方便留下联系方式吗？我让负责人直接和您对接处理。再次为给您带来的不便深表歉意。""",
}

def generate_response(intent, product, domain):
    """根据意图生成回复"""
    template = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["产品咨询"])

    # 填充模板变量
    response = template.format(
        product=product,
        feature1=random.choice(["智能化程度高", "操作简单便捷", "数据安全可靠", "性能稳定", "功能全面"]),
        feature2=random.choice(["7x24小时服务", "专业团队支持", "持续更新迭代", "灵活定制", "快速部署"]),
        feature3=random.choice(["行业领先技术", "丰富的成功案例", "完善的培训体系", "优质的售后保障"]),
        scenario1=random.choice(["日常业务管理", "团队协作", "数据分析", "客户服务"]),
        scenario2=random.choice(["降本增效", "业务拓展", "风险控制", "决策支持"]),
        benefit1=random.choice(["提升工作效率", "降低运营成本", "增加业务收入", "改善用户体验"]),
        benefit2=random.choice(["数据驱动决策", "规范化管理", "提高客户满意度", "增强竞争力"]),
        price1=random.choice(["2,999", "4,999", "6,999", "9,999"]),
        price2=random.choice(["9,999", "19,999", "29,999", "49,999"]),
        gift=random.choice(["专业培训课程", "3个月延保", "增值服务包", "专属顾问服务"]),
        customer_count=random.choice(["500", "1000", "2000", "5000"]),
        renewal_rate=random.choice(["85", "90", "92", "95"]),
        efficiency=random.choice(["30", "40", "50", "60"]),
        roi_months=random.choice(["3", "6", "9", "12"]),
        nps=random.choice(["8.5", "8.8", "9.0", "9.2"]),
        case1=f"某{random.choice(['上市公司', '知名企业', '行业龙头'])}使用后效率提升{random.randint(30, 60)}%",
        case2=f"某{random.choice(['500强企业', '头部品牌', '领军企业'])}已合作{random.randint(2, 5)}年",
        trial_days=random.choice(["7", "14", "15", "30"]),
        promotion=random.choice(["618大促", "年中", "周年庆", "限时"]),
        benefit_time=random.choice(["早用早受益", "抢占先机", "领先竞争对手"]),
        compensation=random.choice(["1个月服务", "200元优惠券", "VIP权益", "专属礼包"]),
        extend_days=random.choice(["30", "60", "90"]),
    )

    return response


def weighted_choice(choices_dict):
    """根据权重随机选择"""
    items = list(choices_dict.keys())
    weights = [choices_dict[k]["weight"] for k in items]
    return random.choices(items, weights=weights, k=1)[0]


def generate_sample():
    """生成单条样本"""
    # 随机选择领域
    domain = random.choice(list(DOMAINS.keys()))
    domain_info = DOMAINS[domain]

    # 随机选择产品
    product = random.choice(domain_info["products"])

    # 根据权重选择意图
    intent = weighted_choice(INTENTS)
    intent_info = INTENTS[intent]

    # 生成问题
    question_template = random.choice(intent_info["questions"])
    competitor = random.choice(COMPETITORS)
    question = question_template.format(product=product, competitor=competitor)

    # 生成回复
    response = generate_response(intent, product, domain)

    # 构造样本
    sample = {
        "messages": [
            {"role": "system", "content": domain_info["system"]},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ],
        # 元数据标签，用于分布分析
        "domain": domain,
        "product": product,
        "intent": intent,
        "difficulty": random.choice(["easy", "medium", "hard"]),
    }

    return sample


def main():
    """生成数据集"""
    samples = []

    for _ in range(1000):
        sample = generate_sample()
        samples.append(sample)

    # 保存为 JSONL
    output_path = "/Users/weixiaochen/Desktop/Tutor/S4/train_platform/datasets/sales_sft.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"生成完成: {len(samples)} 条样本")

    # 统计分布
    print("\n=== 分布统计 ===")

    # 领域分布
    domain_dist = {}
    for s in samples:
        d = s["domain"]
        domain_dist[d] = domain_dist.get(d, 0) + 1
    print("\n领域分布:")
    for k, v in sorted(domain_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(samples)*100:.1f}%)")

    # 意图分布
    intent_dist = {}
    for s in samples:
        i = s["intent"]
        intent_dist[i] = intent_dist.get(i, 0) + 1
    print("\n意图分布:")
    for k, v in sorted(intent_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(samples)*100:.1f}%)")

    # 难度分布
    diff_dist = {}
    for s in samples:
        d = s["difficulty"]
        diff_dist[d] = diff_dist.get(d, 0) + 1
    print("\n难度分布:")
    for k, v in sorted(diff_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(samples)*100:.1f}%)")


if __name__ == "__main__":
    main()
