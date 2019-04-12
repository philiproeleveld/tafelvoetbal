//$(document).ready(function(){

//Find the input search box
//let search = document.getElementsByClassName("form-control")

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
//window.addEventListener('input', function () {
//    let found_elements = search[0].value.trim().toLowerCase()
//    console.log(found_elements)
//    filter(search[0].value.trim().toLowerCase())
//})

//For every word entered by the user, check if the symbol starts with that word
//If it does show the symbol, else hide it
function filter(word) {
    let length = items.length
    let collection = []
    let hidden = 0
    for (let i = 0; i < length; i++) {
    if (items[i].value.toLowerCase().startsWith(word)) {
        $(items[i]).show()
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

