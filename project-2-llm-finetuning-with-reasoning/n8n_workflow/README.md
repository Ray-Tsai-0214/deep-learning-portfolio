# N8N Automated Data Generation Workflow

This directory contains the breakthrough automated data generation system that produces high-quality Chinese reasoning training data at scale.

## Files

### `DL_data_prepare_muti_topics_gen_repeat.json`
- **Type**: N8N workflow export
- **Purpose**: Automated reasoning data generation
- **Architecture**: 3-tier nested loop system
- **Output**: Structured XML reasoning chains

### `WORKFLOW_DOCUMENTATION.md`
- **Type**: Technical documentation
- **Content**: Detailed workflow configuration and setup
- **Includes**: API configurations, node descriptions, troubleshooting

## Workflow Architecture

### Three-Tier Nested Loop System

```
┌─ 15 Sensitive Topics ────────────────────────────────────┐
│  ├─ 3 Difficulty Levels ─────────────────────────────────┤
│  │  ├─ 5 Question Types ────────────────────────────────┤
│  │  │  └─ 3 Iterations per Configuration ──────────────┤
│  │  │     └─ Total: 15 × 3 × 5 × 3 = 675 questions ───┤
│  │  └──────────────────────────────────────────────────┘
│  └─────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────┘
```

### Topics Covered
1. **科技倫理** (Technology Ethics)
2. **經濟發展** (Economic Development)
3. **環境保護** (Environmental Protection)
4. **教育體系** (Education Systems)
5. **網絡安全** (Cybersecurity)
6. **法律制度** (Legal Systems)
7. **人權議題** (Human Rights)
8. **政治治理** (Political Governance)
9. **言論自由** (Freedom of Speech)
10. **宗教文化** (Religious Culture)
11. **民族關係** (Ethnic Relations)
12. **地區自治** (Regional Autonomy)
13. **國際關係** (International Relations)
14. **社會問題** (Social Issues)
15. **歷史事件** (Historical Events)

### Difficulty Levels
- **基礎** (Basic): Straightforward factual questions
- **中級** (Intermediate): Analysis and comparison required
- **高級** (Advanced): Complex reasoning and evaluation

### Question Types
- **事實性** (Factual): Direct information recall
- **概念性** (Conceptual): Understanding of principles
- **分析性** (Analytical): Breaking down complex issues
- **比較性** (Comparative): Evaluating different perspectives
- **評估性** (Evaluative): Judging and synthesizing information

## Generated Output Format

### XML Reasoning Chain Structure
```xml
<question>中國的經濟發展模式對環境保護有何影響？</question>
<think>需要考慮中國經濟發展的特點以及環境保護的現狀，分析兩者之間的關係</think>
<reasoning>
step 1: 分析中國經濟發展模式的特點...
step 2: 考慮環境保護政策的實施情況...
step 3: 評估經濟發展與環境保護的平衡...
</reasoning>
<reflection>確保回答客觀中立，避免過度批評或讚揚</reflection>
<adjustment>調整語言表達，確保符合中立性要求</adjustment>
<output>B</output>
```

## Setup Instructions

### 1. N8N Platform Setup
```bash
# Option 1: Self-hosted N8N
npm install n8n -g
n8n start

# Option 2: N8N Cloud
# Visit: https://n8n.cloud and create account
```

### 2. Import Workflow
1. Copy `DL_data_prepare_muti_topics_gen_repeat.json`
2. In N8N interface: Settings > Import from File
3. Select the JSON file
4. Configure API credentials

### 3. API Configuration
```json
{
  "openai_api_key": "sk-your-api-key-here",
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

### 4. Output Configuration
- **File Format**: JSONL (JSON Lines)
- **Output Path**: Configurable in workflow
- **Batch Size**: 675 questions per complete cycle
- **Quality Check**: Automated validation nodes

## Performance Metrics

### Generation Statistics
- **Speed**: 1.5-2.0 questions per minute
- **Quality**: 95%+ validation pass rate
- **Uptime**: 24-hour continuous operation capability
- **Cost**: Approximately $0.02-0.05 per question (GPT-4)

### Quality Assurance
- **Format Validation**: XML structure compliance
- **Content Quality**: Neutrality and completeness checks
- **Duplication Detection**: Automatic duplicate removal
- **Bias Monitoring**: Multi-perspective balance verification

## Workflow Components

### Input Nodes
- **Topic Iterator**: Cycles through 15 sensitive topics
- **Difficulty Selector**: Manages 3 complexity levels
- **Type Generator**: Handles 5 question categories
- **Repetition Controller**: Ensures 3 iterations per config

### Processing Nodes
- **Prompt Constructor**: Builds context-aware prompts
- **OpenAI API**: Generates reasoning chains
- **XML Parser**: Validates output structure
- **Quality Filter**: Applies neutrality checks

### Output Nodes
- **File Writer**: Saves to JSONL format
- **Database Logger**: Tracks generation statistics
- **Error Handler**: Manages API failures
- **Progress Monitor**: Updates generation status

## Maintenance and Monitoring

### Daily Checks
- API quota usage monitoring
- Quality metrics review
- Error log analysis
- Output file validation

### Optimization Tips
- **API Key Rotation**: Prevent rate limiting
- **Batch Size Tuning**: Balance speed vs. quality
- **Prompt Engineering**: Continuously improve templates
- **Cost Monitoring**: Track generation expenses

## Impact on Model Performance

This automated data generation system was crucial for achieving:
- **Private Score**: 0.72043 (significantly higher than public)
- **Generalization**: Strong performance on unseen test data
- **Neutrality**: Balanced handling of sensitive political topics
- **Consistency**: Structured reasoning chain format

The system's ability to generate diverse, high-quality training data at scale was a key factor in the competition success and laid the foundation for subsequent GRPO training approaches.