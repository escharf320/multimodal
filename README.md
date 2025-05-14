# Multimodal Type Tool

A repository for our multimodal final project. This branch specifically contains the code for the typing tool.

## Installation

The typing tool depends on Node.js and pnpm to run. Instructions for installing both of these libraries are available
[here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) for Node.js and [here](https://pnpm.io/installation) for pnpm.

Our project used Node.js v21.6.2 and pnpm v9.12.2.

```bash
pnpm install # install project dependencies
```

## Usage

To start the Electron application, run the following command. **Note** that when running this command, it will require accessibility features in order to record keystrokes in the background. If an accessibility pops up, make sure to confirm it. Otherwise, be sure to add whatever application you're running this command from (VSCode, Terminal, etc.) to your accessibility settings in MacOS System Preferences.

```bash
pnpm start # start the Electron application
```

## Project Structure

- ### `src/renderer` - Contains desktop application interface files including the streaming QR code

- ### `src/keyLogger.ts` - Helper for starting and stopping the key logger process

- ### `src/main.ts` - The main application command used by Electron to create windows

- ### `src/preload.ts` - A bridge file for setting up communication between the user interface window and the main application backend
