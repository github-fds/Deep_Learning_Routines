@ECHO OFF

RMDIR /S __pycache__

@FOR /D %%I in (*) DO @(
	@IF EXIST %%I\Clean.bat (
		@PUSHD %%I
		@CALL .\Clean.bat
		@POPD
	)
)
