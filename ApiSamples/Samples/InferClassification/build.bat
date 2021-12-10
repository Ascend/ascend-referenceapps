@rem Copyright 2020 Huawei Technologies Co., Ltd
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.

@ECHO OFF

SET BUILD_SCRIPT_PATH=%~dp0
SET BUILD_DIR_PATH=%BUILD_SCRIPT_PATH%\build
SET RELEASE_PATH=%BUILD_SCRIPT_PATH%\dist\release
SET EXE_FULL_PATH=%RELEASE_PATH%\main.exe

rem  Delete the temporary directory generated during the last compilation
IF EXIST %BUILD_DIR_PATH% (RMDIR /S /Q "%BUILD_DIR_PATH%" 2> NUL)
MKDIR "%BUILD_DIR_PATH%"

CD /D %BUILD_DIR_PATH%
rem  Generate the prioject solution by CMakeLists.txt
cmake .. -G "Visual Studio 15 2017 Win64" -DCMAKE_CONFIGURATION_TYPES=release
IF %errorlevel% NEQ 0 (
    ECHO [%date:~0,10%-%time:~0,8%][ERROR]Generate the prioject solution failed, error code: %errorlevel% >> build.log
    EXIT 1
)

rem  Build release version for main.exe
ECHO [%date:~0,10%-%time:~0,8%][INFO]Start to compile main.exe ...  > build.log
cmake --build . --config release >> build.log
IF %errorlevel% NEQ 0 (
    ECHO [%date:~0,10%-%time:~0,8%][ERROR]Build release version for main.exe failed, error code: %errorlevel% >> build.log
    EXIT 2
)

rem  copy config file and model into dist
IF NOT EXIST %EXE_FULL_PATH% (
    ECHO [%date:~0,10%-%time:~0,8%][ERROR]Build failed, dist directory does not exist. >> build.log
    EXIT 3
)

rem  Delete the old data
RMDIR /S /Q "%RELEASE_PATH%\data" 2> NUL
MKDIR "%RELEASE_PATH%\data"
XCOPY "%BUILD_SCRIPT_PATH%\data\config" "%RELEASE_PATH%\data\config" /Y /S /I /E

MKDIR "%RELEASE_PATH%\data\models\resnet"
COPY /Y "%BUILD_SCRIPT_PATH%\data\models\resnet\imagenet1000_clsidx_to_labels.txt" "%RELEASE_PATH%\data\models\resnet"

IF NOT EXIST "%BUILD_SCRIPT_PATH%\data\models\resnet\*.om" (
    ECHO [%date:~0,10%-%time:~0,8%][WARNING] No .om file in %BUILD_SCRIPT_PATH%\data\models\resnet, please convert the model to .om file first. >> build.log
) ELSE (
    COPY /Y "%BUILD_SCRIPT_PATH%\data\models\resnet\*.om" "%RELEASE_PATH%\data\models\resnet"
)

MKDIR "%RELEASE_PATH%\data\models\single_op"

IF NOT EXIST "%BUILD_SCRIPT_PATH%\data\models\single_op\*.om" (
    ECHO [%date:~0,10%-%time:~0,8%][WARNING] No .om file in %BUILD_SCRIPT_PATH%\data\models\single_op, please convert the model to .om file first. >> build.log
) ELSE (
    COPY /Y "%BUILD_SCRIPT_PATH%\data\models\single_op\*.om" "%RELEASE_PATH%\data\models\single_op"
)

ECHO [%date:~0,10%-%time:~0,8%][INFO]Build successfully. >> build.log
EXIT 0