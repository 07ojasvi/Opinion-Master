window.onload = function home() {

    fetch('/check-logged-in', {
        method: 'GET',
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        }
    })
        .then(response => response.json())
        .then(data => {
            apiKey = data.apiKey
        });


    fetch('https://faces.ver.ma/v1/get-person-list', {
        method: 'POST',
        body: JSON.stringify(apiObject),
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        }
    })
        .then(response => response.json())
        .then(data => {


            let personsHTML = document.getElementById("persons")

            for (const persons in data) {

                let person = data[persons]
                const personID = person.personID
                const personName = person.personName
                var thumbnailPath = ""

                if (data.numberOfImages === 0) {
                    thumbnailPath = "https://www.pngitem.com/pimgs/m/30-307416_profile-icon-png-image-free-download-searchpng-employee.png"
                }
                else {
                    thumbnailPath = "https://faces.ver.ma/users/" + apiKey + "/" + personID + "/" + personID + "/" + "_1.jpg"
                }

                personsHTML.innerHTML = `<div class="mt-6 grid grid-cols-1 gap-y-10 gap-x-6 sm:grid-cols-2 lg:grid-cols-4 xl:gap-x-8">
                <div class="group relative">
                    <div class="w-full min-h-80 bg-gray-200 aspect-w-1 aspect-h-1 rounded-md overflow-hidden group-hover:opacity-75 lg:h-80 lg:aspect-none">
                        <img src="${thumbnailPath}" alt="${personName}" class="w-full h-full object-center object-cover lg:w-full lg:h-full">
                    </div>
                <div class="mt-4 flex justify-between">
            <div>
              <h3 class="text-sm text-gray-700">
                <a href="#">
                  <span aria-hidden="true" class="absolute inset-0"></span>
                  ${personName}
                </a>
              </h3>
              
            </div>
            
          </div>
        </div>
      </div>`

            }
        });

}