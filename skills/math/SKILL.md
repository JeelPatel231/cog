---
name: math
description: Evaluate mathematical expressions using a Python script
---

# Math Calculator Skill

## Purpose
This skill executes a Python script to evaluate math expressions.

## When to use
Use this skill ONLY when:
- The user asks for a calculation
- A mathematical expression needs evaluation

## IMPORTANT: How to Execute

DO NOT solve the math yourself.

Instead:
1. Extract the expression from the user input
2. Run the command:

   python calculator.py "<expression>"

3. Return the output from the script

## Example

User input:
"5 + 3 * 2"

Execution:
python calculator.py "5 + 3 * 2"

Output:
11

## Rules
- All the scripts are located in the script folder of the skill
- Always call the script
- Never manually compute the result
- Always pass the full expression as a string