"""
遥感解译助手 v3 — 专业 PDF 报告生成器

基于 reportlab 生成 PDF 格式的专业遥感解译报告。

输出格式:
  - PDF (含封面 / 目录 / 概况 / 方法 / 结果 / 统计 / 采样点 / 结论)

依赖:
  - reportlab>=4.0
  - matplotlib (用于图表渲染)
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ============================================================================
#  中文字体注册
# ============================================================================

def _register_chinese_font():
    """注册中文字体用于 PDF（尝试多个系统路径）。"""
    font_candidates = [
        # Windows
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),
        ("C:/Windows/Fonts/simsun.ttc", "SimSun"),
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
        ("C:/Windows/Fonts/STSONG.TTF", "STSong"),
        ("C:/Windows/Fonts/STKAITI.TTF", "STKaiti"),
        # macOS
        ("/System/Library/Fonts/PingFang.ttc", "PingFang SC"),
        ("/System/Library/Fonts/STHeiti Light.ttc", "Heiti SC"),
        # Linux
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ]
    for fp, name in font_candidates:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("ChineseFont", fp))
                return True
            except Exception:
                continue
    return False


# PDF 字体名称（回退到 Helvetica）
HAS_CJK = _register_chinese_font()
FONT_NAME = "ChineseFont" if HAS_CJK else "Helvetica"
FONT_BOLD = "ChineseFont" if HAS_CJK else "Helvetica-Bold"


# ============================================================================
#  样式定义
# ============================================================================

def _build_styles():
    """构建报告用样式表。"""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        fontName=FONT_BOLD,
        fontSize=28, leading=36,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=HexColor("#1a1a1a"),
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        fontName=FONT_NAME,
        fontSize=14, leading=22,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=HexColor("#555555"),
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        fontName=FONT_BOLD,
        fontSize=18, leading=26,
        spaceBefore=24, spaceAfter=14,
        textColor=HexColor("#2E86AB"),
    ))
    styles.add(ParagraphStyle(
        name="SubSectionTitle",
        fontName=FONT_BOLD,
        fontSize=14, leading=20,
        spaceBefore=14, spaceAfter=8,
        textColor=HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        name="BodyCN",
        fontName=FONT_NAME,
        fontSize=11, leading=18,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName=FONT_BOLD,
        fontSize=10, leading=14,
        alignment=TA_CENTER,
        textColor=white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName=FONT_NAME,
        fontSize=10, leading=14,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="Footer",
        fontName=FONT_NAME,
        fontSize=9, leading=12,
        alignment=TA_CENTER,
        textColor=HexColor("#999999"),
    ))

    return styles


STYLES = _build_styles()


# ============================================================================
#  PDF 报告生成器
# ============================================================================

class PDFReportGenerator:
    """专业 PDF 遥感解译报告生成器。

    Usage:
        gen = PDFReportGenerator(output_dir, "灵川县")
        gen.generate(stats, report_image, sample_points, boundary_info, metadata)
    """

    def __init__(self, output_dir, region_name):
        """初始化生成器。

        Args:
            output_dir: 输出目录路径。
            region_name: 研究区域名称（用于文件名和报告内容）。
        """
        self.output_dir = output_dir
        self.region_name = region_name

    def generate(self, stats, report_image_path=None, sample_points=None,
                 boundary_info=None, metadata=None):
        """生成完整 PDF 报告。

        Args:
            stats: 变化检测统计字典（与 core.py 输出格式一致）。
            report_image_path: 报告 PNG 图片路径（用于插入结果章节）。
            sample_points: 采样点列表 [{"lon":..., "lat":..., "class_name":...}]。
            boundary_info: 边界信息 {"area_km2": ...}。
            metadata: 元信息字典 {
                "region", "year1", "year2", "index_type",
                "data_source", "threshold", "detection_mode",
            }。

        Returns:
            str: 生成的 PDF 文件路径，失败时返回 None。
        """
        meta = metadata or {}
        os.makedirs(self.output_dir, exist_ok=True)

        pdf_path = os.path.join(
            self.output_dir,
            f"{self.region_name}_解译报告.pdf",
        )

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=f"遥感变化解译报告 - {self.region_name}",
            author="遥感解译助手 v3",
        )

        story = []
        self._build_cover(story, meta)
        self._build_toc(story)
        self._build_overview(story, meta, boundary_info)
        self._build_method(story, meta)
        self._build_results(story, stats, meta, report_image_path)
        self._build_sampling(story, sample_points)
        self._build_conclusion(story, stats, meta)

        try:
            doc.build(story)
            return pdf_path
        except Exception as e:
            print(f"  PDF generation failed: {e}")
            return None

    # ---- 封面 ----

    def _build_cover(self, story, meta):
        story.append(Spacer(1, 5 * cm))
        story.append(Paragraph("遥感变化解译报告", STYLES["CoverTitle"]))
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(
            f"{self.region_name}",
            STYLES["CoverSubtitle"],
        ))
        story.append(Paragraph(
            f"{meta.get('year1', '')} → {meta.get('year2', '')}  "
            f"{meta.get('index_type', 'NDVI')}",
            STYLES["CoverSubtitle"],
        ))
        story.append(Spacer(1, 3 * cm))
        story.append(Paragraph(
            f"数据源: {meta.get('data_source', 'Sentinel-2')}",
            STYLES["CoverSubtitle"],
        ))
        story.append(Paragraph(
            f"生成日期: {datetime.now().strftime('%Y-%m-%d')}",
            STYLES["CoverSubtitle"],
        ))
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(
            "遥感解译助手 v3 · 自动生成",
            STYLES["Footer"],
        ))
        story.append(PageBreak())

    # ---- 目录 ----

    def _build_toc(self, story):
        story.append(Paragraph("目录", STYLES["SectionTitle"]))
        toc_items = [
            "1. 项目概况",
            "2. 数据与方法",
            "3. 变化检测结果",
            "4. 统计分析",
            "5. 采样点信息",
            "6. 结论与建议",
        ]
        for item in toc_items:
            story.append(Paragraph(item, STYLES["BodyCN"]))
            story.append(Spacer(1, 0.3 * cm))
        story.append(PageBreak())

    # ---- 项目概况 ----

    def _build_overview(self, story, meta, boundary_info):
        story.append(Paragraph("1. 项目概况", STYLES["SectionTitle"]))

        overview_items = [
            ("研究区域", self.region_name),
            ("前时相", str(meta.get("year1", ""))),
            ("后时相", str(meta.get("year2", ""))),
            ("遥感指数", meta.get("index_type", "NDVI")),
            ("数据源", meta.get("data_source", "Sentinel-2")),
            ("检测模式", meta.get("detection_mode", "multi")),
            ("变化阈值", str(meta.get("threshold", 0.15))),
        ]
        if boundary_info and boundary_info.get("area_km2"):
            overview_items.append(("区域面积", f"{boundary_info['area_km2']} km²"))

        data = [[Paragraph(k, STYLES["TableCell"]),
                 Paragraph(v, STYLES["TableCell"])]
                for k, v in overview_items]
        t = Table(data, colWidths=[4 * cm, 10 * cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#2E86AB")),
            ("TEXTCOLOR", (0, 0), (0, -1), white),
            ("BACKGROUND", (1, 0), (1, -1), HexColor("#F5F5F5")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))
        story.append(PageBreak())

    # ---- 数据与方法 ----

    def _build_method(self, story, meta):
        story.append(Paragraph("2. 数据与方法", STYLES["SectionTitle"]))

        index_type = meta.get("index_type", "NDVI")
        data_source = meta.get("data_source", "Sentinel-2 L2A")

        method_text = (
            f"本报告基于 {data_source} 多光谱卫星影像数据，"
            f"采用 {index_type} 差值分级变化检测方法，"
            "结合连通域斑块面积过滤后处理，"
            "对研究区域进行植被/地物变化分析。"
        )
        story.append(Paragraph(method_text, STYLES["BodyCN"]))
        story.append(Spacer(1, 0.3 * cm))

        if index_type == "NDVI":
            formula = "(NIR - Red) / (NIR + Red)"
            desc = "归一化植被指数，反映植被覆盖度"
        elif index_type == "NDWI":
            formula = "(Green - NIR) / (Green + NIR)"
            desc = "归一化水体指数，反映水体分布"
        elif index_type == "MNDWI":
            formula = "(Green - SWIR1) / (Green + SWIR1)"
            desc = "改进归一化水体指数，抗建筑干扰"
        elif index_type == "NBR":
            formula = "(NIR - SWIR2) / (NIR + SWIR2)"
            desc = "归一化火烧指数，反映火烧迹地"
        elif index_type == "EVI":
            formula = "2.5×(NIR-Red)/(NIR+6×Red-7.5×Blue+1)"
            desc = "增强型植被指数，抗大气干扰"
        elif index_type == "SAVI":
            formula = "(NIR-Red)×(1+L)/(NIR+Red+L)"
            desc = "土壤调节植被指数，低植被覆盖"
        else:
            formula = ""
            desc = ""

        story.append(Paragraph(
            f"<b>指数公式</b>: {index_type} = {formula}",
            STYLES["BodyCN"],
        ))
        story.append(Paragraph(
            f"<b>说明</b>: {desc}",
            STYLES["BodyCN"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        threshold = meta.get("threshold", 0.15)
        story.append(Paragraph(
            f"<b>变化检测参数</b>: 阈值 ±{threshold}，"
            f"后处理方式为连通域逐斑块面积过滤。",
            STYLES["BodyCN"],
        ))
        story.append(PageBreak())

    # ---- 变化检测结果 ----

    def _build_results(self, story, stats, meta, report_image_path):
        story.append(Paragraph("3. 变化检测结果", STYLES["SectionTitle"]))

        # 插入报告图片
        if report_image_path and os.path.exists(report_image_path):
            try:
                img = Image(report_image_path)
                aspect = img.drawWidth / img.drawHeight
                img.drawWidth = 16 * cm
                img.drawHeight = 16 * cm / aspect
                story.append(Spacer(1, 0.5 * cm))
                story.append(img)
                story.append(Spacer(1, 0.3 * cm))
            except Exception:
                story.append(Paragraph("(报告图片加载失败)", STYLES["BodyCN"]))

        story.append(PageBreak())

        # 统计表
        story.append(Paragraph("4. 统计分析", STYLES["SectionTitle"]))

        if not stats or "changed_total" not in stats:
            story.append(Paragraph("无变化检测统计数据。", STYLES["BodyCN"]))
            return

        # 多级统计表
        cls_order = [-3, -2, 1, 2, 3]
        table_data = [
            [Paragraph("类别", STYLES["TableHeader"]),
             Paragraph("面积 (ha)", STYLES["TableHeader"]),
             Paragraph("占比 (%)", STYLES["TableHeader"])],
        ]
        for cv in cls_order:
            s = stats.get(cv)
            if s and s.get("area_ha", 0) > 0:
                table_data.append([
                    Paragraph(s["name"], STYLES["TableCell"]),
                    Paragraph(f"{s['area_ha']:,.0f}", STYLES["TableCell"]),
                    Paragraph(f"{s['percent']:.1f}%", STYLES["TableCell"]),
                ])
        total = stats.get("changed_total", {})
        table_data.append([
            Paragraph("<b>变化合计</b>", STYLES["TableCell"]),
            Paragraph(f"<b>{total.get('area_ha', 0):,.0f}</b>", STYLES["TableCell"]),
            Paragraph(f"<b>{total.get('percent', 0):.1f}%</b>", STYLES["TableCell"]),
        ])

        t = Table(table_data, colWidths=[6 * cm, 4 * cm, 4 * cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2E86AB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [HexColor("#F5F5F5"), white]),
            ("BACKGROUND", (0, -1), (-1, -1), HexColor("#E8F4F8")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

        # 变化构成文字说明
        degrade = sum(
            stats.get(cv, {}).get("area_ha", 0)
            for cv in [-3, -2]
        )
        recover = sum(
            stats.get(cv, {}).get("area_ha", 0)
            for cv in [2, 3]
        )
        story.append(Paragraph(
            f"在 {meta.get('year1','')}—{meta.get('year2','')} 期间，"
            f"<b>{self.region_name}</b> 显著变化总面积 "
            f"<b>{total.get('area_ha', 0):,.0f} ha</b>，"
            f"占总面积的 <b>{total.get('percent', 0):.1f}%</b>。"
            f"其中退化面积 {degrade:,.0f} ha，"
            f"恢复面积 {recover:,.0f} ha。",
            STYLES["BodyCN"],
        ))
        story.append(PageBreak())

    # ---- 采样点信息 ----

    def _build_sampling(self, story, sample_points):
        story.append(Paragraph("5. 采样点信息", STYLES["SectionTitle"]))

        if not sample_points:
            story.append(Paragraph("本次分析未生成采样点。", STYLES["BodyCN"]))
            story.append(PageBreak())
            return

        story.append(Paragraph(
            f"共生成 <b>{len(sample_points)}</b> 个采样点。",
            STYLES["BodyCN"],
        ))
        story.append(Spacer(1, 0.5 * cm))

        # 采样点表（最多显示 30 行）
        max_rows = 30
        pt_data = [
            [Paragraph("序号", STYLES["TableHeader"]),
             Paragraph("经度", STYLES["TableHeader"]),
             Paragraph("纬度", STYLES["TableHeader"]),
             Paragraph("类别", STYLES["TableHeader"])],
        ]
        display_points = sample_points[:max_rows]
        for i, pt in enumerate(display_points, 1):
            pt_data.append([
                Paragraph(str(i), STYLES["TableCell"]),
                Paragraph(f"{pt.get('lon', 0):.5f}", STYLES["TableCell"]),
                Paragraph(f"{pt.get('lat', 0):.5f}", STYLES["TableCell"]),
                Paragraph(pt.get("class_name", ""), STYLES["TableCell"]),
            ])

        if len(sample_points) > max_rows:
            pt_data.append([
                Paragraph("...", STYLES["TableCell"]),
                Paragraph(f"共 {len(sample_points)} 点", STYLES["TableCell"]),
                Paragraph("", STYLES["TableCell"]),
                Paragraph("", STYLES["TableCell"]),
            ])

        t = Table(pt_data, colWidths=[2 * cm, 4 * cm, 4 * cm, 4 * cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2E86AB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [HexColor("#F5F5F5"), white]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(PageBreak())

    # ---- 结论与建议 ----

    def _build_conclusion(self, story, stats, meta):
        story.append(Paragraph("6. 结论与建议", STYLES["SectionTitle"]))

        if not stats or "changed_total" not in stats:
            story.append(Paragraph("无可用统计数据。", STYLES["BodyCN"]))
            return

        total = stats.get("changed_total", {})
        degrade = sum(
            stats.get(cv, {}).get("area_ha", 0)
            for cv in [-3, -2]
        )
        recover = sum(
            stats.get(cv, {}).get("area_ha", 0)
            for cv in [2, 3]
        )

        index_type = meta.get("index_type", "NDVI")

        # 自动生成结论文字
        conclusions = [
            f"在 {meta.get('year1','')}—{meta.get('year2','')} 期间，"
            f"{self.region_name} {index_type} 显著变化总面积 "
            f"{total.get('area_ha', 0):,.0f} 公顷，"
            f"占研究区总面积的 {total.get('percent', 0):.1f}%。",

            f"其中退化面积 {degrade:,.0f} 公顷"
            f"（占变化总面积 {degrade/max(total.get('area_ha',1),1)*100:.1f}%），"
            f"恢复面积 {recover:,.0f} 公顷"
            f"（占变化总面积 {recover/max(total.get('area_ha',1),1)*100:.1f}%）。",
        ]

        # 根据指数类型添加针对性建议
        if index_type in ("NDVI", "EVI", "SAVI"):
            if degrade > recover:
                conclusions.append(
                    "退化面积大于恢复面积，建议关注植被退化区域的生态修复，"
                    "加强水土保持和植被恢复措施。"
                )
            else:
                conclusions.append(
                    "恢复面积大于退化面积，表明植被覆盖状况总体改善，"
                    "建议继续保持现有生态保护措施。"
                )
        elif index_type in ("NDWI", "MNDWI"):
            conclusions.append(
                "建议结合实地调查验证水体变化原因，"
                "区分自然变化与人为活动影响。"
            )
        elif index_type == "NBR":
            conclusions.append(
                "建议对火烧迹地区域进行实地核查，"
                "并根据火烧程度制定差异化的生态恢复方案。"
            )

        for p in conclusions:
            story.append(Paragraph(p, STYLES["BodyCN"]))
            story.append(Spacer(1, 0.2 * cm))

        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            STYLES["Footer"],
        ))
        story.append(Paragraph(
            "遥感解译助手 v3 · 自动生成",
            STYLES["Footer"],
        ))


# ============================================================================
#  便捷函数
# ============================================================================

def generate_report_pdf(stats, report_image=None, sample_points=None,
                        boundary_info=None, metadata=None, output_dir="output"):
    """生成 PDF 解译报告的便捷函数。

    Args:
        stats: 变化检测统计结果。
        report_image: 报告 PNG 图片路径。
        sample_points: 采样点列表。
        boundary_info: 边界信息。
        metadata: 元信息。
        output_dir: 输出目录。

    Returns:
        str: PDF 文件路径，失败返回 None。
    """
    meta = metadata or {}
    region = meta.get("region", "未知区域")
    generator = PDFReportGenerator(output_dir, region)
    return generator.generate(stats, report_image, sample_points,
                              boundary_info, metadata)
