# Project Summary

_Generated with model: gemma-4-e2b-it_

## .gitignore

This directory contains files related to a software project. It includes build artifacts, project context documentation, and the final executable program.

## LICENSE

This file is the MIT License, which grants broad permissions to use, copy, modify, and distribute the software. It includes a copyright notice for fezcode and disclaims all warranties regarding the software. Liability for any damages arising from the software is explicitly excluded.

## README.md

atlas.llm is a local AI coding companion written in Go that provides on-device inference using llama.cpp for coding assistance. It supports multiple modes including an interactive chat, project summarization, semantic code searching, and compiling a full project context into Markdown. Users can manage local models and engines through explicit download commands.

## Recipe.go

This file defines a gobake recipe for building a software project. It sets up a build task that compiles the binary for multiple cross-platform targets including various Linux and Windows architectures. It also includes a clean task to remove the generated build artifacts.

## config.go

This Go file manages the configuration and management of large language models and their execution engines. It defines available models, handles configuration file persistence, and determines the correct file paths for model assets and llama.cpp binaries based on the operating system. The code provides functionality to load a specific model based on the user's configuration.

## dump.go

This Go program is designed to recursively scan a target directory and output the content of files to an output file. It supports configuration options for excluding files and optionally summarizing the content of included files. The program utilizes gitignore rules to skip certain directories and files during the dump process.

