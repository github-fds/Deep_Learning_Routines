@ECHO OFF

RMDIR /S lib
RMDIR /S include

@FOR /D %%I in (*) DO @(
	@IF EXIST %%I\Clean.bat (
		@PUSHD %%I
		@CALL .\Clean.bat
		@POPD
	)
)
