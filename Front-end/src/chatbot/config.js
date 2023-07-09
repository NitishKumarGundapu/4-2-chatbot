import React from "react";
import { createChatBotMessage } from "react-chatbot-kit";

import ChatHead from "./ChatHead"
import Options from "../components/Options";
import Options1 from "../components/Options1";
import Options3 from "../components/Options3";
import Options5 from "../components/Options5";
import Options4 from "../components/Options4";


const config = {
  botName: "Sara",
  initialMessages: [
    createChatBotMessage("Hello, I'm Sara. How may I help you today ? Do you need help with: ", 
    {
      widget: "options",
    }),
  ],
  widgets: 
  [
    {
      widgetName: "options",
      widgetFunc: (props) => <Options {...props} />
    },
    {
      widgetName: "options1",
      widgetFunc: (props) => <Options1 {...props} />
    },
    {
      widgetName: "options5",
      widgetFunc: (props) => <Options5 {...props} />
    },
    {
      widgetName: "options3",
      widgetFunc: (props) => <Options3 {...props} />
    },
    {
      widgetName: "options4",
      widgetFunc: (props) => <Options4 {...props} />
    }
  ],
  customComponents: {
     // Replaces the default header
    header: () => <ChatHead />
    // Replaces the default bot avatar
  },
};

export default config;