@echo off
setlocal enabledelayedexpansion
set val=3
set max_val=35
call conda activate mytorch
for %%I in ( 1280 1024 960 768 640 512 384 256 192 128 64 ) do (
  for %%J in ( 1 2 4 ) do (
    set "sender=%%I %%I %%J% !val!"
    echo python fps.py !sender!
    python fps.py %%I %%I %%J% !val!
    set /a val+=1
    if !val! gtr !max_val! (
      set val=!max_val!
    )
  )
)
pause
endlocal
exit