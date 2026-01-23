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

if "%1"=="" goto interactive

:: Commands
if "%1"=="/help" goto help
if "%1"=="/h" goto help
if "%1"=="/?" goto help
if "%1"=="/i" goto interactive
if "%1"=="/sql" goto sql
if "%1"=="/search" goto search
if "%1"=="/s" goto search
if "%1"=="/stats" goto stats
if "%1"=="/schema" goto schema
if "%1"=="/batch" goto batch
if "%1"=="/b" goto batch
if "%1"=="/export" goto export
if "%1"=="/langgraph" goto langgraph
if "%1"=="/lg" goto langgraph
if "%1"=="/fallback" goto fallback
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

:sql
if "%2"=="" (
    echo %RED%Usage:%RESET% test /sql "SELECT * FROM public.equipment_matrix LIMIT 5"
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

:langgraph
echo.
echo %BOLD%%CYAN%============================================================%RESET%
echo %BOLD%  LangGraph Agent Test%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo %YELLOW%Testing LangGraph agent with sample query...%RESET%
echo.
set "USE_LANGGRAPH_AGENT=true"
if "%2"=="" (
    python cli_tester.py "Wie viele Kettenfertiger haben wir?"
) else (
    python cli_tester.py "%~2"
)
goto end

:fallback
echo.
echo %BOLD%%YELLOW%============================================================%RESET%
echo %BOLD%  Fallback Agent Test (LangGraph Disabled)%RESET%
echo %YELLOW%============================================================%RESET%
echo.
echo %YELLOW%Testing with LangGraph DISABLED...%RESET%
echo.
set "USE_LANGGRAPH_AGENT=false"
if "%2"=="" (
    python cli_tester.py "Wie viele Kettenfertiger haben wir?"
) else (
    python cli_tester.py "%~2"
)
goto end

:validate
echo.
echo %BOLD%%GREEN%============================================================%RESET%
echo %BOLD%  LangGraph Migration Validation%RESET%
echo %GREEN%============================================================%RESET%
echo.
python -c "print('=== Import Validation ==='); from rag.langgraph_agent import LangGraphAgent, execute_sql, search_documents, find_columns, explore_column, SYSTEM_PROMPT; print('[OK] All LangGraph imports successful'); from rag.search import RAGSearch; print('[OK] RAGSearch import successful'); from rag.config import config; print(f'[OK] USE_LANGGRAPH_AGENT={config.use_langgraph_agent}'); print(); print('=== Tool Validation ==='); tools = [execute_sql, search_documents, find_columns, explore_column]; [print(f'[OK] {t.name}: invoke={hasattr(t,\"invoke\")}, ainvoke={hasattr(t,\"ainvoke\")}') for t in tools]; print(); print('=== All Validations Passed ===')"
goto end

:help
echo.
echo %BOLD%%CYAN%============================================================%RESET%
echo %BOLD%  Teams Bot RAG Testing System%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo %BOLD%Usage:%RESET% test [query] or test /command [args]
echo.
echo %BOLD%Commands:%RESET%
echo   %GREEN%/help%RESET%           Show this help
echo   %GREEN%/i%RESET%              Interactive mode
echo   %GREEN%/sql "query"%RESET%    Direct SQL query
echo   %GREEN%/search "text"%RESET%  Document search (Pinecone)
echo   %GREEN%/stats%RESET%          Database statistics
echo   %GREEN%/schema%RESET%         Schema information
echo   %GREEN%/batch%RESET%          Run batch tests
echo   %GREEN%/export%RESET%         Export results to JSON
echo.
echo %BOLD%LangGraph Commands:%RESET%
echo   %GREEN%/langgraph%RESET%      Test with LangGraph agent (or /lg)
echo   %GREEN%/fallback%RESET%       Test with LangGraph disabled
echo   %GREEN%/validate%RESET%       Validate LangGraph migration
echo.
echo %BOLD%Examples:%RESET%
echo   %YELLOW%test Wie viele Bomag Maschinen haben wir?%RESET%
echo   %YELLOW%test /sql "SELECT COUNT(*) FROM sema_matrix.equipment_matrix"%RESET%
echo   %YELLOW%test /search "Anleitung Kaltfraese"%RESET%
echo   %YELLOW%test /lg "Zeige alle Voegele Kettenfertiger"%RESET%
echo   %YELLOW%test /validate%RESET%
echo.
goto end

:end
endlocal
