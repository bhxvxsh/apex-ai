@echo off
REM APEX AI - Quick Launcher for APEX Terminal CLI
REM Runs the Claude Code-style interactive command-line interface

cd /d "%~dp0"
py -3.12 apex_cli.py %*

