#!/usr/bin/env python3
"""
销售领域数据集生成脚本
生成 SFT, DPO, GRPO 格式的训练数据
"""

import json
import random
import os

# 销售场景模板库
INDUSTRIES = {
    "B2B软件": {
        "system": "你是一位专业的B2B软件销售顾问，擅长企业软件销售，能够准确把握客户需求并提供专业的解决方案。",
        "products": ["CRM系统", "ERP系统", "OA办公系统", "数据分析平台", "云存储服务", "网络安全解决方案", "协同办公软件", "人力资源管理系统", "财务管理软件", "项目管理工具"],
        "scenarios": ["价格谈判", "功能咨询", "竞品对比", "技术支持", "合同签订", "售后服务", "升级续费", "定制开发", "培训需求", "方案汇报"]
    },
    "房地产": {
        "system": "你是一位专业的房产销售顾问，熟悉本地房产市场，能够为客户提供专业的购房建议和置业规划。",
        "products": ["住宅公寓", "别墅", "商铺", "写字楼", "公寓", "二手房", "学区房", "海景房", "地铁房", "精装房"],
        "scenarios": ["首次咨询", "看房预约", "价格协商", "贷款咨询", "合同条款", "交房验收", "投资建议", "户型推荐", "周边配套", "政策解读"]
    },
    "汽车销售": {
        "system": "你是一位专业的汽车销售顾问，熟悉各品牌车型配置和性能，能够根据客户需求推荐最合适的车型。",
        "products": ["轿车", "SUV", "MPV", "新能源车", "混合动力车", "跑车", "皮卡", "商务车", "越野车", "小型车"],
        "scenarios": ["车型咨询", "试驾预约", "价格谈判", "金融方案", "旧车置换", "保险咨询", "售后保养", "提车流程", "加装配置", "比较竞品"]
    },
    "保险": {
        "system": "你是一位专业的保险顾问，具备丰富的保险产品知识，能够为客户提供全面的风险保障规划。",
        "products": ["人寿保险", "医疗保险", "意外险", "重疾险", "车险", "家财险", "养老保险", "教育金", "企业险", "旅游险"],
        "scenarios": ["产品咨询", "保障规划", "理赔流程", "保费计算", "保单变更", "续保提醒", "方案对比", "家庭保障", "投保流程", "健康告知"]
    },
    "教育培训": {
        "system": "你是一位专业的教育咨询顾问，了解各类培训课程和教育资源，能够为学员提供个性化的学习规划。",
        "products": ["语言培训", "职业技能", "学历提升", "考研辅导", "留学咨询", "IT培训", "艺术培训", "资格认证", "企业内训", "线上课程"],
        "scenarios": ["课程咨询", "学习规划", "师资了解", "费用咨询", "试听安排", "报名流程", "效果保障", "课程对比", "就业服务", "退费政策"]
    },
    "家电零售": {
        "system": "你是一位专业的家电销售顾问，熟悉各类家电产品的功能特点和使用技巧，能够为客户推荐最适合的产品。",
        "products": ["电视", "冰箱", "洗衣机", "空调", "热水器", "油烟机", "微波炉", "扫地机器人", "空气净化器", "净水器"],
        "scenarios": ["产品咨询", "型号对比", "价格优惠", "安装服务", "售后保修", "能耗对比", "尺寸匹配", "品牌推荐", "以旧换新", "配送安排"]
    },
    "金融理财": {
        "system": "你是一位专业的理财顾问，具备丰富的金融产品知识，能够为客户提供专业的资产配置建议。",
        "products": ["基金定投", "银行理财", "股票投资", "债券", "黄金", "外汇", "保本理财", "信托产品", "私募基金", "存款产品"],
        "scenarios": ["产品咨询", "风险评估", "收益分析", "投资组合", "赎回流程", "市场分析", "定期复盘", "资产配置", "税务规划", "退休规划"]
    },
    "医疗健康": {
        "system": "你是一位专业的健康顾问，熟悉各类医疗健康服务和产品，能够为客户提供专业的健康管理建议。",
        "products": ["体检套餐", "健康管理", "康复服务", "医疗器械", "营养保健品", "中医理疗", "心理咨询", "牙科服务", "眼科服务", "美容医疗"],
        "scenarios": ["服务咨询", "套餐推荐", "预约流程", "费用说明", "报告解读", "随访服务", "方案定制", "会员权益", "医生介绍", "注意事项"]
    },
    "旅游服务": {
        "system": "你是一位专业的旅游顾问，熟悉国内外旅游目的地和线路，能够为客户提供个性化的旅行规划。",
        "products": ["跟团游", "自由行", "定制游", "邮轮游", "亲子游", "蜜月游", "商务考察", "签证服务", "机票酒店", "景点门票"],
        "scenarios": ["目的地咨询", "行程规划", "价格咨询", "签证办理", "酒店推荐", "交通安排", "保险建议", "特殊需求", "退改政策", "行程调整"]
    },
    "电信服务": {
        "system": "你是一位专业的电信业务顾问，熟悉各类通信产品和套餐，能够为客户提供最优的通信解决方案。",
        "products": ["手机套餐", "宽带服务", "流量包", "企业通信", "物联网卡", "国际漫游", "固定电话", "云服务", "安全服务", "增值业务"],
        "scenarios": ["套餐咨询", "资费对比", "携号转网", "宽带安装", "故障报修", "账单查询", "合约变更", "企业需求", "优惠活动", "服务投诉"]
    }
}

# 客户提问模板
CUSTOMER_QUESTIONS = {
    "价格谈判": [
        "价格能优惠点吗？",
        "有没有折扣？",
        "太贵了，能便宜点吗？",
        "预算有限，最低多少钱？",
        "付款方式可以分期吗？",
        "批量采购有优惠吗？",
        "现在有什么促销活动？",
        "这个价格包含什么服务？",
        "竞争对手报价更低，你们能匹配吗？",
        "首付可以少交一点吗？"
    ],
    "功能咨询": [
        "这个产品有什么功能？",
        "和其他产品有什么区别？",
        "能满足我的需求吗？",
        "有哪些规格可选？",
        "支持定制吗？",
        "有试用版吗？",
        "使用起来复杂吗？",
        "有什么限制条件？",
        "后续还会更新功能吗？",
        "能详细介绍一下核心功能吗？"
    ],
    "售后服务": [
        "售后服务怎么样？",
        "保修期多长？",
        "出问题了怎么处理？",
        "有客服支持吗？",
        "退换货政策是什么？",
        "维修费用怎么算？",
        "服务响应时间多长？",
        "有上门服务吗？",
        "培训服务包含吗？",
        "售后电话是多少？"
    ],
    "竞品对比": [
        "你们和XX品牌比有什么优势？",
        "市面上这么多选择，为什么选你们？",
        "听说XX产品也不错，你怎么看？",
        "你们的市场占有率是多少？",
        "有什么独特的竞争优势？",
        "你们的客户都是哪些？",
        "和行业龙头比差在哪里？",
        "技术上有什么领先之处？",
        "服务上有什么差异化？",
        "品牌知名度怎么样？"
    ],
    "决策犹豫": [
        "我需要再考虑一下",
        "我要和家人商量",
        "我需要向领导汇报",
        "能给我一些时间吗？",
        "我还在比较其他选项",
        "现在不是购买的好时机",
        "我担心买了会后悔",
        "你们公司靠谱吗？",
        "买了之后效果不好怎么办？",
        "这个决定太重大了"
    ],
    "使用问题": [
        "这个怎么使用？",
        "操作起来难吗？",
        "需要培训吗？",
        "有使用说明吗？",
        "遇到问题怎么办？",
        "有人指导吗？",
        "上手需要多长时间？",
        "有没有视频教程？",
        "常见问题有哪些？",
        "出错了怎么处理？"
    ],
    "信任建立": [
        "你们公司成立多久了？",
        "有成功案例吗？",
        "能提供客户参考吗？",
        "有什么资质认证？",
        "团队背景怎么样？",
        "有没有负面评价？",
        "合作的大客户有哪些？",
        "媒体报道过吗？",
        "获得过什么奖项？",
        "有行业认可吗？"
    ],
    "紧急需求": [
        "能加急处理吗？",
        "最快什么时候能用？",
        "可以今天就办理吗？",
        "能优先安排吗？",
        "有现货吗？",
        "能提前交付吗？",
        "加急需要额外费用吗？",
        "周末可以办理吗？",
        "在线就能完成吗？",
        "多长时间能搞定？"
    ]
}

# 高质量回复模板元素
RESPONSE_ELEMENTS = {
    "开场": [
        "感谢您的咨询！",
        "您好！很高兴为您服务。",
        "感谢您对我们的关注！",
        "您好！这是一个很好的问题。",
        "感谢您的信任！",
        "您好！让我为您详细解答。",
        "非常感谢您的垂询！",
        "您好！我来为您介绍一下。"
    ],
    "共情": [
        "完全理解您的顾虑，",
        "我非常理解您的心情，",
        "您的担心是很正常的，",
        "这确实是大家都关心的问题，",
        "换做是我也会有同样的考虑，",
        "您考虑得非常周全，",
        "这个问题问得很好，",
        "您的谨慎态度是对的，"
    ],
    "结尾": [
        "您看这样可以吗？",
        "如有其他问题随时联系我。",
        "期待与您进一步沟通。",
        "您觉得怎么样？",
        "我们可以约个时间详谈。",
        "有任何疑问请随时告诉我。",
        "希望能帮到您！",
        "期待您的回复！"
    ]
}

# 低质量回复特征
BAD_RESPONSE_PATTERNS = [
    "不知道",
    "这个我说了不算",
    "你自己看吧",
    "就这样",
    "没办法",
    "不可能",
    "你去问别人",
    "随便你",
    "爱买不买",
    "我很忙"
]


def generate_sft_sample(industry_name, industry_data):
    """生成单条SFT样本"""
    product = random.choice(industry_data["products"])
    scenario_type = random.choice(list(CUSTOMER_QUESTIONS.keys()))
    question = random.choice(CUSTOMER_QUESTIONS[scenario_type])

    # 个性化问题
    question = question.replace("XX", random.choice(["竞品A", "竞品B", "市场领先品牌", "其他供应商"]))

    # 构建具体的问题上下文
    context_additions = [
        f"我想了解一下你们的{product}，",
        f"关于{product}，",
        f"我正在考虑{product}，",
        f"听说你们{product}不错，",
        f"我需要{product}，",
    ]

    full_question = random.choice(context_additions) + question

    # 生成高质量回复
    response = generate_quality_response(industry_name, product, scenario_type, question)

    return {
        "messages": [
            {"role": "system", "content": industry_data["system"]},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": response}
        ]
    }


def generate_quality_response(industry, product, scenario_type, question):
    """生成高质量销售回复"""
    opening = random.choice(RESPONSE_ELEMENTS["开场"])
    empathy = random.choice(RESPONSE_ELEMENTS["共情"])
    closing = random.choice(RESPONSE_ELEMENTS["结尾"])

    # 根据场景生成核心内容
    if scenario_type == "价格谈判":
        body = f"""
{empathy}在价格方面我们确实有一定的灵活空间。

**针对您的情况，我有以下方案建议：**

1. **标准方案**：我们的{product}标准配置价格合理，能满足大部分需求
2. **优惠方案**：如果您今天能确定，我可以申请额外的折扣
3. **分期方案**：支持分期付款，首付比例灵活可谈

**额外价值：**
- 首年免费维护服务
- 专属客户经理一对一支持
- 定期产品培训和更新

我建议我们可以坐下来详细沟通您的具体需求，我来为您定制一个最具性价比的方案。"""

    elif scenario_type == "功能咨询":
        body = f"""
让我为您详细介绍{product}的核心功能：

**主要功能亮点：**

1. **核心功能**
   - 行业领先的技术架构
   - 智能化操作界面
   - 强大的数据处理能力

2. **差异化优势**
   - 相比同类产品效率提升30%以上
   - 独特的AI智能辅助功能
   - 完善的安全保障机制

3. **扩展能力**
   - 支持与主流系统集成
   - 灵活的定制开发接口
   - 持续的功能更新迭代

**适用场景：**
- 适合各种规模的用户使用
- 特别适合追求效率和品质的客户

我可以为您安排一次详细的产品演示，让您亲身体验这些功能。"""

    elif scenario_type == "售后服务":
        body = f"""
我们非常重视售后服务，这是我们的核心竞争力之一。

**售后服务体系：**

1. **响应承诺**
   - 7×24小时客服热线
   - 2小时内响应，24小时内解决
   - 专属客户经理全程跟进

2. **保障政策**
   - 行业领先的质保期
   - 免费软件更新和升级
   - 定期巡检和维护服务

3. **增值服务**
   - 免费产品培训
   - 定期使用优化建议
   - VIP客户专属活动

4. **无忧保障**
   - 不满意可退换
   - 全国联保服务网络
   - 配件终身供应承诺

我们有很多老客户用了多年，售后体验是他们选择复购的重要原因。"""

    elif scenario_type == "竞品对比":
        body = f"""
{empathy}选择之前多做比较是非常明智的。

**客观分析我们的优势：**

1. **产品层面**
   - 核心技术自主研发，持续迭代优化
   - 功能覆盖更全面，特别是在{industry}领域
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
   - 服务过XX+家客户
   - 客户续约率超过90%
   - 多个行业标杆案例

当然每个品牌都有其特点，我建议您可以实际体验后再做决定，我们可以安排一次对比演示。"""

    elif scenario_type == "决策犹豫":
        body = f"""
{empathy}做重要决策确实需要慎重考虑。

**我的建议是：**

1. **先明确核心需求**
   - 您目前最需要解决的问题是什么？
   - 有哪些是必须满足的条件？
   - 预算范围大概是多少？

2. **可以分步进行**
   - 先试用体验一段时间
   - 小范围试点验证效果
   - 逐步扩大使用范围

3. **我们的保障**
   - 提供免费试用期
   - 不满意全额退款承诺
   - 专业团队协助评估

4. **资料支持**
   - 我可以准备详细的方案资料
   - 提供同类客户的参考案例
   - 安排与现有客户交流

不急于决定，您充分了解清楚再做选择。我这边随时配合您的节奏。"""

    elif scenario_type == "使用问题":
        body = f"""
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

大部分用户1-2天就能熟练使用，如果您愿意，我现在就可以给您演示一下基本操作。"""

    elif scenario_type == "信任建立":
        body = f"""
{empathy}选择合作伙伴确实需要了解清楚。

**关于我们：**

1. **公司实力**
   - 成立X年，专注{industry}领域
   - 服务客户数量XX+
   - 年营收稳步增长

2. **资质认证**
   - 具备相关行业资质
   - 通过ISO质量管理认证
   - 获得多项技术专利

3. **客户认可**
   - 客户满意度持续90%以上
   - 续约率行业领先
   - 多次获得行业奖项

4. **案例参考**
   - 可以提供同行业客户参考
   - 安排客户证言和案例分享
   - 实地参观我们的运营中心

我可以发送我们的公司介绍和客户案例集给您，也欢迎您来公司参观交流。"""

    else:  # 紧急需求
        body = f"""
{empathy}我完全理解时间的紧迫性。

**针对加急需求：**

1. **快速响应**
   - 立即启动绿色通道
   - 优先安排专人处理
   - 最大程度压缩流程时间

2. **具体时间**
   - 标准流程：X个工作日
   - 加急处理：可缩短至X天
   - 特急情况：当天可完成基础部分

3. **配合事项**
   - 请您提前准备相关材料
   - 保持电话畅通便于对接
   - 配合线上审核流程

4. **服务保障**
   - 加急不影响服务质量
   - 全程进度实时同步
   - 问题第一时间沟通

请告诉我您具体需要在什么时间前完成，我来评估一下可行性并制定方案。"""

    return f"{opening}\n{body}\n\n{closing}"


def generate_bad_response(industry, product, scenario_type):
    """生成低质量回复（用于DPO的rejected）"""
    bad_patterns = [
        f"这个{product}就是这样的，没什么特别的。",
        f"价格就是这个价格，不能改。",
        "你自己上网查一下吧。",
        f"我也不太清楚{product}的具体情况。",
        "这个要问其他同事，我不负责这块。",
        "没有优惠，原价销售。",
        "你考虑好了再来吧。",
        f"这个{product}反正挺好的，买就对了。",
        "售后的事情以后再说。",
        "我很忙，你有什么问题？",
        "就这些功能，不能定制。",
        "不知道，你问售后。",
        "保修期过了就要收费的。",
        "其他品牌我不了解，反正我们的最好。",
        f"你预算不够就别考虑{product}了。",
    ]
    return random.choice(bad_patterns)


def generate_dpo_sample(industry_name, industry_data):
    """生成单条DPO样本"""
    product = random.choice(industry_data["products"])
    scenario_type = random.choice(list(CUSTOMER_QUESTIONS.keys()))
    question = random.choice(CUSTOMER_QUESTIONS[scenario_type])

    context_additions = [
        f"我想了解一下你们的{product}，",
        f"关于{product}，",
        f"我正在考虑{product}，",
    ]

    full_question = random.choice(context_additions) + question
    prompt = f"系统设定：{industry_data['system']}\n\n客户问题：{full_question}"

    chosen = generate_quality_response(industry_name, product, scenario_type, question)
    rejected = generate_bad_response(industry_name, product, scenario_type)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


def generate_grpo_sample(industry_name, industry_data):
    """生成单条GRPO样本"""
    product = random.choice(industry_data["products"])
    scenario_type = random.choice(list(CUSTOMER_QUESTIONS.keys()))
    question = random.choice(CUSTOMER_QUESTIONS[scenario_type])

    context_additions = [
        f"我想了解一下你们的{product}，",
        f"关于{product}，",
        f"我正在考虑{product}，",
    ]

    full_question = random.choice(context_additions) + question
    prompt = f"系统设定：{industry_data['system']}\n\n客户问题：{full_question}"

    # 生成不同质量的回复
    quality_level = random.choice(["high", "medium", "low"])

    if quality_level == "high":
        response = generate_quality_response(industry_name, product, scenario_type, question)
        reward_signals = {
            "professionalism": round(random.uniform(0.85, 1.0), 2),
            "empathy": round(random.uniform(0.80, 1.0), 2),
            "solution_oriented": round(random.uniform(0.85, 1.0), 2),
            "customer_retention": round(random.uniform(0.80, 1.0), 2),
            "overall": round(random.uniform(0.85, 0.98), 2)
        }
    elif quality_level == "medium":
        response = generate_medium_response(industry_name, product, scenario_type)
        reward_signals = {
            "professionalism": round(random.uniform(0.5, 0.75), 2),
            "empathy": round(random.uniform(0.4, 0.7), 2),
            "solution_oriented": round(random.uniform(0.5, 0.75), 2),
            "customer_retention": round(random.uniform(0.45, 0.7), 2),
            "overall": round(random.uniform(0.5, 0.72), 2)
        }
    else:
        response = generate_bad_response(industry_name, product, scenario_type)
        reward_signals = {
            "professionalism": round(random.uniform(0.1, 0.35), 2),
            "empathy": round(random.uniform(0.05, 0.3), 2),
            "solution_oriented": round(random.uniform(0.1, 0.35), 2),
            "customer_retention": round(random.uniform(0.05, 0.25), 2),
            "overall": round(random.uniform(0.1, 0.3), 2)
        }

    return {
        "prompt": prompt,
        "response": response,
        "reward_signals": reward_signals
    }


def generate_medium_response(industry, product, scenario_type):
    """生成中等质量回复"""
    medium_patterns = [
        f"您好，{product}的基本情况是这样的：功能还可以，价格中等，您可以考虑一下。",
        f"关于{product}，我简单说一下：有几个不同的版本可以选，具体看您的需求吧。",
        f"{product}确实不错，很多客户都在用。您主要想了解哪方面？",
        f"这个问题嘛，{product}的情况和市场上其他产品差不多，有兴趣可以详细聊聊。",
        f"好的，{product}目前有优惠活动，您可以先了解一下基本信息。",
        f"嗯，您说的这个问题，我们的{product}是可以解决的。需要我详细介绍吗？",
        f"{product}的售后服务还行，有问题可以联系客服处理。",
        f"价格方面可以商量，您的预算大概是多少？我看看有没有合适的方案。",
    ]
    return random.choice(medium_patterns)


def main():
    """主函数：生成数据集"""
    output_dir = os.path.dirname(os.path.abspath(__file__))

    sft_file = os.path.join(output_dir, "sales_sft.jsonl")
    dpo_file = os.path.join(output_dir, "sales_dpo.jsonl")
    grpo_file = os.path.join(output_dir, "sales_grpo.jsonl")

    # 目标数量
    target_sft = 1000
    target_dpo = 1000
    target_grpo = 1000

    print(f"开始生成数据集...")
    print(f"目标：SFT {target_sft}条, DPO {target_dpo}条, GRPO {target_grpo}条")

    # 生成SFT数据
    print("\n生成SFT数据...")
    sft_samples = []
    industries_list = list(INDUSTRIES.items())
    for i in range(target_sft):
        industry_name, industry_data = random.choice(industries_list)
        sample = generate_sft_sample(industry_name, industry_data)
        sft_samples.append(sample)
        if (i + 1) % 100 == 0:
            print(f"  SFT: {i + 1}/{target_sft}")

    with open(sft_file, 'w', encoding='utf-8') as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"SFT数据已保存: {sft_file} ({len(sft_samples)}条)")

    # 生成DPO数据
    print("\n生成DPO数据...")
    dpo_samples = []
    for i in range(target_dpo):
        industry_name, industry_data = random.choice(industries_list)
        sample = generate_dpo_sample(industry_name, industry_data)
        dpo_samples.append(sample)
        if (i + 1) % 100 == 0:
            print(f"  DPO: {i + 1}/{target_dpo}")

    with open(dpo_file, 'w', encoding='utf-8') as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"DPO数据已保存: {dpo_file} ({len(dpo_samples)}条)")

    # 生成GRPO数据
    print("\n生成GRPO数据...")
    grpo_samples = []
    for i in range(target_grpo):
        industry_name, industry_data = random.choice(industries_list)
        sample = generate_grpo_sample(industry_name, industry_data)
        grpo_samples.append(sample)
        if (i + 1) % 100 == 0:
            print(f"  GRPO: {i + 1}/{target_grpo}")

    with open(grpo_file, 'w', encoding='utf-8') as f:
        for sample in grpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"GRPO数据已保存: {grpo_file} ({len(grpo_samples)}条)")

    print("\n数据集生成完成！")
    print(f"  - {sft_file}: {len(sft_samples)}条")
    print(f"  - {dpo_file}: {len(dpo_samples)}条")
    print(f"  - {grpo_file}: {len(grpo_samples)}条")


if __name__ == "__main__":
    main()
