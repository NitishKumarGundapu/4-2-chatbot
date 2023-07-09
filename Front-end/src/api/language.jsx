
class language {
    constructor() {
        this.language = "english"
    }

    getLanguage() {
        return this.language
    }

    setLanguage(lang) {
        if (lang === 1){
        this.language = "telugu"
        }
        if (lang === 2){
        this.language = "hindi"
            }
        if (lang === 3){
            this.language = "english"
        }
        if (lang === 4){
            this.language = "tamil"
        }
        if (lang === 5){
            this.language = "kannada"
        }
    }
}

//export default language

var a = new language()

export default a