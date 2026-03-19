# Course Project Specification

**Course:** DTU 02214 — Hardware/Software Codesign (Spring 2026)

## Task

Create a working machine learning-based computer vision application on the **XIAO ESP32-S3 Sense**.

**Our choice:** A face detection application that recognizes the faces of the team members but rejects all other faces.

> Alternative: propose another computer vision application (must be approved by teachers). The task should be demanding enough to run into performance constraints on the ESP32-S3 and require trade-offs.

Write and hand in a **report** covering the application, dataset, design space, evaluation, test, and code.

Teams can be **two or three students**.

---

## Application Requirements

- Must run on the **XIAO ESP32-S3 Sense** board
- Must use the **onboard camera** and a **machine learning model** to visually recognize one or more objects (including living beings)
- Must fulfill your stated requirements for **accuracy, latency, and other performance specs** (you choose the numbers)
- The ML model must be **at least partly trained by you** on **data collected by you** (transfer learning is allowed)
- Training and model conversion pipeline must be written in **Python**
- ESP32-S3 application must be written in **C/C++**

---

## Report Requirements

The report must contain at least:

1. **Application description** — purpose, scope, requirements, and how it works
2. **Dataset description** — classes, conditional factors, amount of data, sources, and data format
3. **Design and implementation**
   - Training, conversion, and optimization pipeline overview (can optionally be provided as markdown files with the code)
   - ESP32 application code overview (can optionally be provided as markdown files with the code)
   - Presentation of the **design space** and its parameters
   - Justification of **design choices and trade-offs** within the design space
4. **Verification**
   - Model performance evaluation including relevant performance metrics
   - Real-world test results
5. **Team member contributions** — account of each member's contribution to the project and report

**Length:** 10–30 pages (excluding code).

---

## Deliverables

| Deliverable | Format |
|-------------|--------|
| Training + application code | Zip file (no generated files — only Python/C/C++ and config files) **or** link to a public git repo |
| Written report | PDF |

---

## Evaluation Criteria

- Fulfillment of application and report requirements
- How well design decisions are **motivated and justified using course theory**
- How well the implementation reflects **methods and best practices** from the course
- Overall quality and completeness of the report

---

## Timeline

| Date | Milestone |
|------|-----------|
| **February 26** (week 4) | Project registration deadline |
| **May 7** (week 13) | Report and code delivery deadline |
