import React from "react";
import api from "../api/retriver"
import a from "../api/language"
import "./main.css";
import {useState,useEffect} from "react";

const text1 = [
  "1. Home Eye Test",
  "2. Try On"
]

const Options4 = (props) => 
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

  const options = [
    {
      text: text[0],
      handler: props.actionProvider.home_eye_test,
      id: 1,
    },
    { 
      text: text[1], 
      handler: props.actionProvider.try_on,
      id: 2,
    }
  ];

  const buttonsMarkup = options.map((option) => (
    <button key={option.id} onClick={option.handler} className="option-button">
      {option.text}
    </button>
  ));

  return <div className="options-container">{buttonsMarkup}</div>;
};

export default Options4;
