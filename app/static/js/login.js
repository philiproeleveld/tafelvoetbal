//$(document).ready(function(){

//Find the input search box
let search = document.getElementsByClassName("form-control")

//Find every item inside the dropdown
let items = document.getElementsByClassName("dropdown-item")
function buildDropDown(values) {
    let contents = []
    for (let name of values) {
    name = name.replace(/\b\w/g, l => l.toUpperCase())
    contents.push('<input type="button" class="dropdown-item" type="button" value="' + name + '"/>')
    }
    $('.menuItems').append(contents.join(""))

    //Hide the row that shows no items were found
    $('.empty').hide()
}

//Capture the event when user types into the search box
window.addEventListener('input', function () {
    let found_elements = search[0].value.trim().toLowerCase()
    console.log(found_elements)
    filter(search[0].value.trim().toLowerCase()) // inputs search letters to filter (i.e. ti)
})

//For every word entered by the user, check if the symbol starts with that word
//If it does show the symbol, else hide it
function filter(word) {
    let length = items.length // lengte van usernames
    let collection = []
    let hidden = 0

    // Loop over all items in the items array
    for (let i = 0; i < length; i++) {
        // Show if there is some sort of match
        if (items[i].value.toLowerCase().startsWith(word)) {
            let word = items[i].value
            console.log(word)
            let selected_values = $('.hidden-form')
//            console.log()
            let selected = false

            selected_values.each(function(input_element) {
                if (input_element.value == word) {
                    selected = true
                    console.log(input_element.value)
                }
            })

            if (selected == true) {
                $(items[i]).show()
            } else {
                $(items[i]).hide()
                hidden++
            }
        }
        else {
            $(items[i]).hide()
            hidden++
        }
    }

    //If all items are hidden, show the empty view
    if (hidden === length) {
    $('.empty').show()
    }
    else {
    $('.empty').hide()
    }
}

//If the user clicks on any item, set the title of the button as the text of the item
$('.menuItems').on('click', '.dropdown-item', function(){

    // Set title of button
    let dropdown_sibling = $(this).parent().parent().siblings(".dropdown-toggle")
    dropdown_sibling.text($(this)[0].value)
    dropdown_sibling.dropdown('toggle')

    // Also set the value of hidden form attribute to the selected value
    let post_sibling = $(this).parent().parent().siblings(".hidden-form")
    post_sibling.val($(this)[0].value)
})

buildDropDown(names)

//})

