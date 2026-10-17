# 遥感变化解译报告

**{{ region_name }}**  ·  {{ year1 }} → {{ year2 }}

---

## 1. 项目概况

| 项目 | 内容 |
|------|------|
| 研究区域 | {{ region_name }} |
| 分析时相 | {{ year1 }} 年 → {{ year2 }} 年 |
| 数据源 | {{ data_source }} |
| 指数类型 | {{ index_type }} |
| 检测模式 | {{ detection_mode }} |
| 变化阈值 | ±{{ threshold }} |

---

## 2. 数据与方法

- **数据源**: {{ data_source_desc }}
- **方法**: 基于 {{ index_type }} 差值的多级变化检测 + 连通域斑块过滤
- **空间分辨率**: 10 m (Sentinel-2) / 30 m (Landsat)

---

## 3. 变化检测结果

### 3.1 分类统计

| 类别 | 面积 (ha) | 占比 |
|------|-----------|------|
| 严重退化 | {{ severe_degrade_ha }} | {{ severe_degrade_pct }}% |
| 中度退化 | {{ moderate_degrade_ha }} | {{ moderate_degrade_pct }}% |
| 稳定 | {{ stable_ha }} | {{ stable_pct }}% |
| 中度恢复 | {{ moderate_recover_ha }} | {{ moderate_recover_pct }}% |
| 强烈恢复 | {{ severe_recover_ha }} | {{ severe_recover_pct }}% |
| **变化合计** | **{{ total_change_ha }}** | **{{ total_change_pct }}%** |

### 3.2 变化分布图

![变化检测结果]({{ report_image }})

---

## 4. 采样点信息

共生成 **{{ num_sample_points }}** 个采样点。

采样点文件: `{{ sample_file }}`

---

## 5. 结论

在 {{ year1 }}—{{ year2 }} 期间，{{ region_name }} 显著变化总面积为 **{{ total_change_ha }} ha**，
占总面积的 **{{ total_change_pct }}%**。

其中:
- 退化面积: {{ degrade_ha }} ha
- 恢复面积: {{ recover_ha }} ha

---

*报告生成时间: {{ report_time }}*
*遥感解译助手 v3*
