
import a from "../api/language"
import api from "../api/retriver"
import {useNavigate} from 'react-router-dom'
import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';
import 'firebase/compat/auth';
import 'firebase/compat/analytics';

//import { useCollectionData } from 'react-firebase-hooks/firestore';

firebase.initializeApp({
  apiKey: "AIzaSyCjw7uSk1JHtymgBPRDWJs2W0EB2vIaB0A",
  authDomain: "project-7f27d.firebaseapp.com",
  projectId: "project-7f27d",
  storageBucket: "project-7f27d.appspot.com",
  messagingSenderId: "845652508520",
  appId: "1:845652508520:web:f1dc91daf2cd3dd9f1e0dd",
  measurementId: "G-9YLWFK2X5K"
})

const firestore = firebase.firestore();
const messagesRef = firestore.collection('messages');


const text1 = [
  "1. Steps to place an order online or offline stores",
  "2. Finding nearest Lenskart Store in your surroundings",
  "3. Cancellation of Ordered Items with order number",
  "4. Making an appointment for a home eye test and try-on",
  "5. Select different input and display language",
  "6. Logout from your lenskart account"
]
class ActionProvider  {
    constructor(createChatBotMessage, setStateFunc ,createClientMessage) {
      this.createChatBotMessage = createChatBotMessage;
      this.createClientMessage = createClientMessage;
      this.setState = setStateFunc;
      this.ids = ["100011","100012","100013","100014","100015"]
      this.order_id = ""
      this.navigate = useNavigate()
    }


      get_translation = async (message,language) => {
        const response = await api.get("/translater",{
          params : {
          msg: message,
          lang: language
        }})
        return response
      };

      get_response = async (message,language) => {
        const response = await api.get("/get_response",{
          params : {
          msg: message,
          lang: language
        }})
        return response
      };

    start_again = () => {
      this.get_translation("Hello, I'm Sara. How may I help you today ? Do you need help with: ",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data, {widget: "options",});
        this.addMessageToState1(message)
        await messagesRef.add({
          text: "start again",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    warning_1 = (lowercase) => {
      this.get_response(lowercase,a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data);
        this.addMessageToState(message)
        await messagesRef.add({
          text: lowercase,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };


    /*------------------------------------------------Options-----------------------------------------------*/ 
    
    starting_option_1 = () => {
      this.get_translation(text1[0],a.getLanguage()).then(async (response) => {
        const message = this.createClientMessage(response.data, {widget: "options1",});
        this.addMessageToState(message);
        await messagesRef.add({
          text: response.data,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    starting_option_2 = () => {
      var lang_str =  ''.concat([text1[1], "Please enter your area pincode number"])
      this.get_translation(lang_str,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(",")
        var message = this.createClientMessage(msg[0])
        this.addMessageToState(message);
        message = this.createChatBotMessage(msg[1]);
        this.addMessageToState(message);
        await messagesRef.add({
          text: response.data,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    starting_option_3 = () => {
      var lang_str =  ''.concat([text1[2], "Please enter the order-id for your order"])
      this.get_translation(lang_str,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(",")
        var message = this.createClientMessage(msg[0]);
        this.addMessageToState(message);
        message = this.createChatBotMessage(msg[1]);
        this.addMessageToState(message);
        await messagesRef.add({
          text: response.data,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    starting_option_4 = () => {
      var lang_str =  ''.concat([text1[3], "Please select on of 2 options"])
      this.get_translation(lang_str,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(",")
        var message = this.createClientMessage(msg[0]);
        this.addMessageToState(message);
        message = this.createChatBotMessage(msg[1], {widget: "options4",});
        this.addMessageToState(message);
        await messagesRef.add({
          text: response.data,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    starting_option_5 = () => {
      var lang_str =  ''.concat([text1[4], "Please Select any below Language"])
      this.get_translation(lang_str,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(",")
        var message = this.createClientMessage(msg[0]);
        this.addMessageToState(message);
        message = this.createChatBotMessage(msg[1], {widget: "options5",});
        this.addMessageToState(message);
        await messagesRef.add({
          text: response.data,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    starting_option_6 = () => {
      this.get_translation(text1[5],a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data);
        this.addMessageToState(message)
        a.setLanguage(3);
        await messagesRef.add({
          text: response.data,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
        this.navigate('/')
      })
    };

   /*------------------------------------------------Option 1-----------------------------------------------*/ 

    place_an_order_online = () => {
      let request_string = "1.Place an order online ; Please contact our nearby store and make an order or please contact to our customer care 2014587400 or visit store.lenskart.com for details "
      this.get_translation(request_string,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(";")
        var message = this.createClientMessage(msg[0]);
        this.addMessageToState(message)
        message = this.createChatBotMessage(msg[1]);
        this.addMessageToState(message)
        await messagesRef.add({
          text: "Place an order online",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

    place_an_order_offline = () => {
      let request_string = "2.Place an order offline ; Please contact our nearby store and make an order or please contact to our customer care 2014587400 or visit store.lenskart.com for details "
      this.get_translation(request_string,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(";")
        var message = this.createClientMessage(msg[0]);
        this.addMessageToState(message)
        message = this.createChatBotMessage(msg[1]);
        this.addMessageToState(message)
        await messagesRef.add({
          text: "Place an order offline",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    };

     /*------------------------------------------------Option 2-----------------------------------------------*/ 

     locate_store = (lowercase) => {
      //const lowercase = getPincode();
      const get_translation = async (message,language) => {
        const response = await api.get("/address",{
          params : {
            pincode: message,
          lang: language
        }})
        return response
      };
      get_translation(lowercase,a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        await messagesRef.add({
          text: lowercase,
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    } 

    /*------------------------------------------------Option 3-----------------------------------------------*/
    
    cancel_order = (lowercase) => {
      if (this.ids.includes(lowercase)) {
        this.order_id = lowercase
        this.get_translation("Do you want to cancel the order , type yes or no",a.getLanguage()).then(async (response) => {
          const message = this.createChatBotMessage(response.data,{widget: "options3",})
          this.addMessageToState(message)
          await messagesRef.add({
            text: lowercase,
            createdAt: firebase.firestore.FieldValue.serverTimestamp()
          })
        })
      }
      else{
        this.get_translation("Please enter the correct order-id ",a.getLanguage()).then(async (response) => {
          const message = this.createChatBotMessage(response.data);
          this.addMessageToState(message);
        })
      }
    }
    cancel_order_yes = () => {
      var index = this.ids.indexOf(this.order_id)
      delete this.ids[index];
      this.get_translation("Order deleted Sucessfully",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        await messagesRef.add({
          text: "yes",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    cancel_order_no = () => {
      this.get_translation("Cancellation Process terminated Sucessfully",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        await messagesRef.add({
          text: "no",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    /*------------------------------------------------Option 4-----------------------------------------------*/ 

    home_eye_test = () => {
      var lang_str =  ''.concat(["1. Home eye test", "Please contact our Customer service centre to get an appointment - contact 180-100-1111"])
      this.get_translation(lang_str,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(",")
        var message = this.createClientMessage(msg[0])
        this.addMessageToState(message)
        message = this.createChatBotMessage(msg[1])
        this.addMessageToState(message)
        await messagesRef.add({
          text: "home eye test",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    try_on = () => {
      var lang_str =  ''.concat(["2. Try on", "This feature is available in android or the ios app , please download the app to see the feature"])
      this.get_translation(lang_str,a.getLanguage()).then(async (response) => {
        const msg = response.data.split(",")
        var message = this.createClientMessage(msg[0])
        this.addMessageToState(message)
        message = this.createChatBotMessage(msg[1])
        this.addMessageToState(message)
        await messagesRef.add({
          text: "try on",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    /*------------------------------------------------Option 5-----------------------------------------------*/ 

    language_option_1 = () => {
      a.setLanguage(1);
      console.log(a.getLanguage())
      this.get_translation("Language changed to telugu",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        this.start_again()
        await messagesRef.add({
          text: "telugu",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    language_option_2 = () => {
      a.setLanguage(2);
      console.log(a.getLanguage())
      this.get_translation("Language changed to hindi",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        this.start_again()
        await messagesRef.add({
          text: "hindi",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    language_option_3 = () => {
      a.setLanguage(3);
      console.log(a.getLanguage())
      this.get_translation("Language changed to english",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        this.start_again()
        await messagesRef.add({
          text: "english",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    language_option_4 = () => {
      a.setLanguage(4);
      console.log(a.getLanguage())
      this.get_translation("Language changed to tamil",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        this.start_again()
        await messagesRef.add({
          text: "tamil",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    language_option_5 = () => {
      a.setLanguage(5);
      console.log(a.getLanguage())
      this.get_translation("Language changed to kannada",a.getLanguage()).then(async (response) => {
        const message = this.createChatBotMessage(response.data)
        this.addMessageToState(message)
        this.start_again()
        await messagesRef.add({
          text: "kanadda",
          createdAt: firebase.firestore.FieldValue.serverTimestamp()
        })
      })
    }

    addMessageToState = (message) => {
      this.setState((prevState) => ({
        ...prevState,
        messages: [...prevState.messages, message],
      }));
    };

    addMessageToState1 = (message) => {
      this.setState((prevState) => ({
        ...prevState,
        messages: [message],
      }));
    };
  }
  
  export default ActionProvider;
