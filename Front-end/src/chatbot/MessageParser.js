class MessageParser {
    constructor(actionProvider) {
      this.actionProvider = actionProvider;
    }

    parse = (message) => {
      //console.log(message);
      const lowercase = message.toLowerCase();

      if(lowercase.includes("start")){
        this.actionProvider.start_again();
      }

      else if(lowercase.includes("5000")){
        this.actionProvider.locate_store(lowercase);
      }

      else if(lowercase.includes("1000")){
        this.actionProvider.cancel_order(lowercase);
      }
      else{
        this.actionProvider.warning_1(lowercase);
      }
    }
  
    parser = (message) => {
      //console.log(message);
      const lowercase = message.toLowerCase();

      if(lowercase.includes("start")){
        this.actionProvider.start_again();
      }

      if(lowercase.includes("5000")){
        this.actionProvider.locate_store(lowercase);
      }

      if(lowercase.includes("1000")){
        this.actionProvider.cancel_order(lowercase);
      }

      if (lowercase.includes("yes")){
        this.actionProvider.cancel_order_yes();
      }

      if (lowercase.includes("no")){
        this.actionProvider.cancel_order_no();
      }
    }
  }
  
  export default MessageParser;