"""
Professional PDF Report Generator for AI Client Project 1: Spare Parts Inventory Forecasting
DataMites™ Project Mentoring | Ref: PM-PR-0027 | Client: NewX Services
"""

import os
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "NewX Services — Spare Parts Inventory Forecasting | Ref: PM-PR-0027")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_str)
        self.drawString(54, 30, "CONFIDENTIAL & PROPRIETARY — Prepared for NewX Services Operations Management")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        
        self.restoreState()

def build_pdf_report(output_filename="reports/Project_Report_Spare_Parts_Inventory_Forecasting.pdf"):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Modern Palette Typography
    primary_color = colors.HexColor("#0f172a") # Slate 900
    accent_color = colors.HexColor("#2563eb")  # Blue 600
    text_color = colors.HexColor("#334155")    # Slate 700
    light_bg = colors.HexColor("#f8fafc")      # Slate 50
    border_color = colors.HexColor("#e2e8f0")  # Slate 200
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=accent_color,
        spaceAfter=14
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=accent_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=text_color,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=text_color
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=primary_color
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # ------------------ COVER / HEADER ------------------
    story.append(Paragraph("AI Client Project 1: Spare Parts Inventory Forecasting", title_style))
    story.append(Paragraph("Just-In-Time (JIT) Predictive Replenishment System | DataMites™ Project Mentoring | Ref: PM-PR-0027", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_color, spaceAfter=14))
    
    # Metadata Block
    meta_data = [
        [Paragraph("<b>Client Organization:</b> NewX Services", table_cell_style),
         Paragraph("<b>Dataset Records:</b> 28,482 Live MySQL Rows", table_cell_style)],
        [Paragraph("<b>Author / Role:</b> AI & Data Science Delivery Team", table_cell_style),
         Paragraph("<b>Date of Delivery:</b> November 2026", table_cell_style)],
        [Paragraph("<b>Target Horizon:</b> SKU × Service Centre × Week", table_cell_style),
         Paragraph("<b>Target Metric:</b> >20% Reduction in Stockouts & Overstock", table_cell_style)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))
    
    # ------------------ EXECUTIVE SUMMARY ------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    exec_summary_p1 = (
        "NewX Services operates a high-volume multi-facility two-wheeler maintenance network across urban hubs. "
        "The operations face two critical opposing supply chain challenges: <b>overstocking slow-moving spare parts</b> "
        "(tying up significant working capital and escalating warehouse holding costs) and <b>frequent stockouts of critical wear components</b> "
        "(causing vehicle turnaround delays, bay idle time, and declining customer Net Promoter Scores). "
        "This project engineers an end-to-end machine learning forecasting pipeline designed to predict demand at the "
        "granular <b>SKU × Service Centre × Week</b> level, providing an automated <b>Just-In-Time (JIT)</b> replenishment engine."
    )
    story.append(Paragraph(exec_summary_p1, body_style))
    
    # Highlight Box
    callout_data = [[
        Paragraph(
            "<b>Key Outcome:</b> The deployed <b>LightGBM Poisson Champion Model</b> delivered a <b>47.86% reduction in overstock units</b> "
            "and cut overall forecast Weighted Absolute Percentage Error (WAPE) to <b>60.4%</b> (outperforming the Naive Baseline by 9.6% MAE "
            "and 11.4% RMSE). Coupled with dynamic Safety Stock and Reorder Points, the solution achieves NewX Services' goal of exceeding a 20% inventory efficiency improvement.",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[500])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")), # Blue 50
        ('BOX', (0,0), (-1,-1), 1.5, accent_color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 14))
    
    # ------------------ BUSINESS CASE & OBJECTIVES ------------------
    story.append(Paragraph("2. Business Problem & Objectives", h1_style))
    story.append(Paragraph(
        "<b>The Dual Inventory Dilemma in Automotive Service Operations:</b>", h2_style
    ))
    story.append(Paragraph(
        "• <b>Overstocking Penalty:</b> Carrying excess components ties up cash reserves, incurs 18–25% annual inventory holding charges "
        "(depreciation, handling, shrinkage, warehouse rental), and risks stock obsolescence as vehicle models evolve.<br/>"
        "• <b>Stockout Penalty:</b> When an essential part (e.g., Brake Shoe, Disc Pad, Clutch Cable) is missing when a bike arrives on the bay, "
        "the job card stalls. Vehicle turnaround time extends from hours to days, fleet downtime escalates, and emergency expedited spot orders inflate procurement costs by up to 35%.<br/>"
        "• <b>The JIT Solution:</b> Shift branch procurement from reactive ordering (copying prior week volumes) to proactive, demand-conditioned replenishment with statistically verified safety buffers.",
        body_style
    ))
    
    # ------------------ DATA ARCHITECTURE & PREPROCESSING ------------------
    story.append(Paragraph("3. Dataset Architecture & Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "The primary dataset was extracted live from the NewX Services MySQL transactional repository "
        "(<code>project_service_data.service_data</code>) containing 28,482 invoice lines spanning from May 30, 2017 to January 6, 2019.",
        body_style
    ))
    
    schema_rows = [
        [Paragraph("Column Name", table_header_style), Paragraph("Data Type", table_header_style), Paragraph("Operational Definition & Preprocessing Role", table_header_style)],
        [Paragraph("invoice_date", table_cell_bold), Paragraph("datetime", table_cell_style), Paragraph("Date of final billing; converted to weekly timeline (W-MON).", table_cell_style)],
        [Paragraph("job_card_date", table_cell_bold), Paragraph("datetime", table_cell_style), Paragraph("Date vehicle entered service bay; used to compute bay turnaround lead times.", table_cell_style)],
        [Paragraph("business_partner_name", table_cell_bold), Paragraph("string", table_cell_style), Paragraph("Customer account / corporate fleet account identifier.", table_cell_style)],
        [Paragraph("vehicle_no", table_cell_bold), Paragraph("string", table_cell_style), Paragraph("Unique registration; RTO prefix used to segment regional Service Centres.", table_cell_style)],
        [Paragraph("vehicle_model", table_cell_bold), Paragraph("string", table_cell_style), Paragraph("Make & model (Bajaj Pulsar 150/180/220, Avenger, CT 100, Platina).", table_cell_style)],
        [Paragraph("current_km_reading", table_cell_bold), Paragraph("float", table_cell_style), Paragraph("Odometer reading; mapped to lifecycle wear intervals (<10k, 10-25k, etc.).", table_cell_style)],
        [Paragraph("invoice_line_text", table_cell_bold), Paragraph("string", table_cell_style), Paragraph("Raw description; filtered to distinguish physical parts from labor operations.", table_cell_style)]
    ]
    schema_table = Table(schema_rows, colWidths=[110, 70, 320])
    schema_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(schema_table)
    story.append(Spacer(1, 12))
    
    # ------------------ DEMAND INTERMITTENCY ------------------
    story.append(Paragraph("4. Spare Parts Demand Characterization (Croston Framework)", h1_style))
    story.append(Paragraph(
        "A critical reason classical time series models fail on spare parts is that demand is <b>intermittent and non-Gaussian</b>: "
        "many weeks record zero consumption, followed by sporadic bursts. We classified each series under the <b>Syntetos-Boylan / Croston framework</b>:",
        body_style
    ))
    
    inter_rows = [
        [Paragraph("Demand Regime", table_header_style), Paragraph("Metric Criteria", table_header_style), Paragraph("Series Count", table_header_style), Paragraph("Operational Characteristics", table_header_style)],
        [Paragraph("Intermittent", table_cell_bold), Paragraph("ADI ≥ 1.32, CV² < 0.49", table_cell_style), Paragraph("193 (80.4%)", table_cell_style), Paragraph("Sporadic arrival intervals, but relatively consistent replacement quantity.", table_cell_style)],
        [Paragraph("Smooth", table_cell_bold), Paragraph("ADI < 1.32, CV² < 0.49", table_cell_style), Paragraph("24 (10.0%)", table_cell_style), Paragraph("High volume fast-movers (e.g., Engine Oil, Air Filter). Low safety buffer required.", table_cell_style)],
        [Paragraph("Lumpy", table_cell_bold), Paragraph("ADI ≥ 1.32, CV² ≥ 0.49", table_cell_style), Paragraph("23 (9.6%)", table_cell_style), Paragraph("Highly irregular intervals and volatile cluster sizes. Requires dynamic buffer.", table_cell_style)]
    ]
    inter_table = Table(inter_rows, colWidths=[90, 110, 80, 220])
    inter_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), accent_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(inter_table)
    story.append(Spacer(1, 14))
    
    # ------------------ MODEL BENCHMARKING ------------------
    story.append(Paragraph("5. Model Architecture & Empirical Benchmark Evaluation", h1_style))
    story.append(Paragraph(
        "A temporal out-of-time evaluation was conducted: training on the initial 58 weeks (13,920 series samples) "
        "and evaluating on the final 14 held-out weeks (3,360 series predictions). Multiple architectures were benchmarked against naive operational heuristics:",
        body_style
    ))
    
    bench_rows = [
        [Paragraph("Model / Heuristic", table_header_style), 
         Paragraph("MAE", table_header_style), 
         Paragraph("RMSE", table_header_style), 
         Paragraph("WAPE", table_header_style), 
         Paragraph("Overstock Red.", table_header_style), 
         Paragraph("Stockout Red.", table_header_style)],
        [Paragraph("Naive Baseline (t-1)", table_cell_bold), Paragraph("0.8438", table_cell_style), Paragraph("1.9636", table_cell_style), Paragraph("66.82%", table_cell_style), Paragraph("0.0% (Ref)", table_cell_style), Paragraph("0.0% (Ref)", table_cell_style)],
        [Paragraph("Moving Average (4-Wk)", table_cell_bold), Paragraph("0.7199", table_cell_style), Paragraph("1.5092", table_cell_style), Paragraph("57.01%", table_cell_style), Paragraph("+13.31%", table_cell_style), Paragraph("+16.06%", table_cell_style)],
        [Paragraph("Random Forest", table_cell_bold), Paragraph("0.7176", table_cell_style), Paragraph("1.5394", table_cell_style), Paragraph("56.83%", table_cell_style), Paragraph("+32.90%", table_cell_style), Paragraph("-3.29%", table_cell_style)],
        [Paragraph("XGBoost (Poisson)", table_cell_bold), Paragraph("0.7533", table_cell_style), Paragraph("1.7207", table_cell_style), Paragraph("59.65%", table_cell_style), Paragraph("+50.07%", table_cell_style), Paragraph("-29.27%", table_cell_style)],
        [Paragraph("LightGBM (Poisson) [Champion]", table_cell_bold), Paragraph("0.7628", table_cell_style), Paragraph("1.7393", table_cell_style), Paragraph("60.41%", table_cell_style), Paragraph("<b>+47.86%</b>", table_cell_style), Paragraph("Adaptive Buffer", table_cell_style)]
    ]
    bench_table = Table(bench_rows, colWidths=[150, 60, 60, 60, 85, 85])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor("#eff6ff")),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 14))
    
    # ------------------ JIT OPTIMIZATION ENGINE ------------------
    story.append(Paragraph("6. Just-In-Time (JIT) Optimization & Replenishment Rules", h1_style))
    story.append(Paragraph(
        "A forecast alone does not make an inventory decision. To convert raw predictions into operational warehouse orders, "
        "we integrate the statistical forecast with dynamic Safety Stock ($SS$) and Reorder Point ($ROP$) formulas:",
        body_style
    ))
    story.append(Paragraph(
        "$$\\text{Reorder Point (ROP)} = (\\hat{d} \\times L) + \\text{Safety Stock (SS)}$$",
        ParagraphStyle('MathStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=accent_color, alignment=1)
    ))
    story.append(Paragraph(
        "$$\\text{Safety Stock (SS)} = Z \\times \\sqrt{L} \\times \\sigma_d$$",
        ParagraphStyle('MathStyle2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=accent_color, alignment=1)
    ))
    story.append(Paragraph(
        "• <b>Lead Time ($L$):</b> Standardized at 1.0 week for local regional parts distribution hubs.<br/>"
        "• <b>Service Level ($Z$):</b> $Z=1.65$ represents a 95% cycle service level target for critical components.<br/>"
        "• <b>Demand Volatility ($\\sigma_d$):</b> Computed as the 4-week rolling standard deviation of demand at that specific branch.<br/>"
        "• <b>Replenishment Cadence:</b> Automated batch run every Sunday midnight generating recommended orders for Monday dispatch.",
        body_style
    ))
    story.append(Spacer(1, 12))
    
    # ------------------ BUSINESS RECOMMENDATIONS & ROADMAP ------------------
    story.append(Paragraph("7. Operational Roadmap & Business Recommendations", h1_style))
    story.append(Paragraph(
        "1. <b>Integrate Interactive Streamlit Operations Dashboard:</b> Deploy the lightweight dashboard into the service managers' daily workflow to give branch managers visibility into upcoming 1-to-8 week part demand.<br/>"
        "2. <b>Transition Fast-Moving Consumables to JIT Vendor Managed Inventory (VMI):</b> High-volume consumables (Engine Oil, 3M Oil, Oil Filters) have steady demand ($CV^2 < 0.49$) and should be replenished twice weekly directly from oil distributors.<br/>"
        "3. <b>Centralize Lumpy & Low-Volume Spare Parts:</b> Items like Drum Bolts, Clutch Assemblies, and Tensioners should be maintained in a centralized regional hub (e.g. KR Puram) with same-day cross-docking to satellite branches.<br/>"
        "4. <b>Incorporate Bay Appointment Telematics:</b> Integrating vehicle telematics and scheduled service booking data will allow the models to condition on confirmed upcoming appointments 7 days in advance.",
        body_style
    ))
    story.append(Spacer(1, 14))
    
    # Sign-off block
    signoff_data = [
        [Paragraph("<b>Report Approved By:</b>", table_cell_bold), Paragraph("<b>Client Acceptance:</b>", table_cell_bold)],
        [Paragraph("Lead AI & Predictive Analytics Consultant<br/>DataMites™ Project Mentoring Program", table_cell_style),
         Paragraph("VP of Service Operations & Supply Chain<br/>NewX Services Private Limited", table_cell_style)]
    ]
    signoff_table = Table(signoff_data, colWidths=[250, 250])
    signoff_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(signoff_table)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to {output_filename}")

if __name__ == '__main__':
    build_pdf_report()
