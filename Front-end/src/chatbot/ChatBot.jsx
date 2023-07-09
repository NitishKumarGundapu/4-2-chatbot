import React from "react";

import config from "./config";
import ActionProvider from "./ActionProvider";
import MessageParser from "./MessageParser";

import Chatbot from "react-chatbot-kit";
//import 'react-chatbot-kit/build/main.css';

import './main.css'

function ChatBot() {
    return (
      <div className="App">
        <div>
          <Chatbot config={config} actionProvider={ActionProvider} messageParser={MessageParser} />
        </div>
      </div>
    );
  }

export default ChatBot