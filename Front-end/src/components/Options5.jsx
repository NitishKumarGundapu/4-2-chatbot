import React from "react";
import api from "../api/retriver"
import a from "../api/language"
import "./main.css";
import {useState,useEffect} from "react";

const text1 = [
  "1. Telugu",
  "2. Hindi",
  "3. English",
  "4. Tamil",
  "5. kannada"
]

const Options5 = (props) => 
{
  const [text,setText] = useState({})  

  const get_translation = async (message,language) => {
    const response = await api.get("/translation",{
      params : {
      msg: ''.concat(message),
      lang: language
    }})
    return setText(response.data);
  };

  useEffect(() => {get_translation(text1,a.getLanguage())},[]);

  useEffect(() => {
    
}, []);

  const options = [
    {
      text: text[0],
      handler: props.actionProvider.language_option_1,
      id: 1,
    },
    { 
      text: text[1], 
      handler: props.actionProvider.language_option_2,
      id: 2,
    },
    {
      text: text[2],
      handler: props.actionProvider.language_option_3,
      id: 3,
    },
    {
      text: text[3],
      handler: props.actionProvider.language_option_4,
      id: 4,
    },
    {
      text: text[4],
      handler: props.actionProvider.language_option_5,
      id: 5,
    }
  ];

  const buttonsMarkup = options.map((option) => (
    <button key={option.id} onClick={option.handler} className="option-button">
      {option.text}
    </button>
  ));

  return <div className="options-container">{buttonsMarkup}</div>;
};

export default Options5;
