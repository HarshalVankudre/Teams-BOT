@echo off
setlocal enabledelayedexpansion

:: Enable ANSI colors (Windows 10+)
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

:: Colors
set "GREEN=%ESC%[32m"
set "CYAN=%ESC%[36m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[31m"
set "BOLD=%ESC%[1m"
set "RESET=%ESC%[0m"

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: LangGraph is the only agent now
set "USE_LANGGRAPH_AGENT=true"

if "%1"=="" goto interactive

:: Commands
if "%1"=="/help" goto help
if "%1"=="/h" goto help
if "%1"=="/?" goto help
if "%1"=="/i" goto interactive
if "%1"=="/v" goto verbose_query
if "%1"=="/verbose" goto verbose_query
if "%1"=="/sql" goto sql
if "%1"=="/search" goto search
if "%1"=="/s" goto search
if "%1"=="/stats" goto stats
if "%1"=="/schema" goto schema
if "%1"=="/batch" goto batch
if "%1"=="/b" goto batch
if "%1"=="/export" goto export
if "%1"=="/validate" goto validate

:: Plain text = query
goto query

:interactive
echo.
echo %BOLD%%CYAN%============================================================%RESET%
echo %BOLD%  Teams Bot Interactive Testing%RESET%
echo %CYAN%============================================================%RESET%
python cli_tester.py -i
goto end

:query
set "QUERY=%*"
echo.
echo %YELLOW%Query:%RESET% %QUERY%
echo.
python cli_tester.py "%QUERY%"
goto end

:verbose_query
:: Shift to get the query after /v
shift
set "QUERY=%1 %2 %3 %4 %5 %6 %7 %8 %9"
echo.
echo %CYAN%[Verbose Mode]%RESET%
echo %YELLOW%Query:%RESET% %QUERY%
echo.
python cli_tester.py -v "%QUERY%"
goto end

:sql
if "%2"=="" (
    echo %RED%Usage:%RESET% test /sql "SELECT * FROM public.equipment_matrix_v2 LIMIT 5"
    goto end
)
set "SQL_QUERY=%~2"
echo %CYAN%SQL:%RESET% %SQL_QUERY%
python cli_tester.py -s "%SQL_QUERY%"
goto end

:search
if "%2"=="" (
    echo %RED%Usage:%RESET% test /search "search query"
    goto end
)
set "SEARCH_QUERY=%~2"
echo %CYAN%Search:%RESET% %SEARCH_QUERY%
python cli_tester.py -d "%SEARCH_QUERY%"
goto end

:stats
echo %CYAN%Database Statistics%RESET%
python cli_tester.py --stats
goto end

:schema
echo %CYAN%Schema Information%RESET%
python cli_tester.py --schema
goto end

:batch
echo %YELLOW%Running batch tests...%RESET%
python cli_tester.py -b
goto end

:export
echo %GREEN%Exporting results...%RESET%
python cli_tester.py -o test_results.json
goto end

:validate
echo.
echo %BOLD%%GREEN%============================================================%RESET%
echo %BOLD%  LangGraph Agent Validation%RESET%
echo %GREEN%============================================================%RESET%
echo.
python -c "print('=== Import Validation ==='); from rag.langgraph_agent import LangGraphAgent, get_langgraph_agent, set_shared_postgres, set_shared_pinecone; print('[OK] LangGraph runtime imports successful'); from rag.langgraph_tools import get_langgraph_tools; print(f'[OK] Tool registry loaded: {len(get_langgraph_tools())} tools'); from rag.prompts import LANGGRAPH_SYSTEM_PROMPT; print('[OK] Prompt import successful'); from rag.search import RAGSearch; print('[OK] RAGSearch import successful'); from rag.config import config; print(f'[OK] Config loaded'); print(f'[OK] LangGraph model: {config.langgraph_model}'); print(); print('=== All Validations Passed ===')"
goto end

:help
echo.
echo %BOLD%%CYAN%============================================================%RESET%
echo %BOLD%  Teams Bot RAG Testing System (Simplified)%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo %BOLD%Usage:%RESET% test [query] or test /command [args]
echo.
echo %BOLD%Commands:%RESET%
echo   %GREEN%/help%RESET%           Show this help
echo   %GREEN%/i%RESET%              Interactive mode
echo   %GREEN%/v "query"%RESET%      Verbose mode - shows tool calls and details
echo   %GREEN%/sql "query"%RESET%    Direct SQL query
echo   %GREEN%/search "text"%RESET%  Document search (Pinecone)
echo   %GREEN%/stats%RESET%          Database statistics
echo   %GREEN%/schema%RESET%         Schema information
echo   %GREEN%/batch%RESET%          Run batch tests
echo   %GREEN%/export%RESET%         Export results to JSON
echo   %GREEN%/validate%RESET%       Validate LangGraph agent
echo.
echo %BOLD%Examples:%RESET%
echo   %YELLOW%test Wie viele Bomag Maschinen haben wir?%RESET%
echo   %YELLOW%test /v Fertiger mit 2m Einbaubreite%RESET%
echo   %YELLOW%test /sql "SELECT COUNT(*) FROM public.equipment_matrix_v2"%RESET%
echo   %YELLOW%test /search "Anleitung Kaltfraese"%RESET%
echo   %YELLOW%test /validate%RESET%
echo.
goto end

:end
endlocal
