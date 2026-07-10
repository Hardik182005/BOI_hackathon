#!/usr/bin/env bash
cd "$(dirname "$0")/../frontend"
npm test --silent && npm run build --silent
