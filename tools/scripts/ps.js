// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


const debug = require('debug')('pubsub');


function createRandomStr(length) {
    var mask = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~`!@#$%^&*()_+-={}[]:";\'<>?,./|\\';
    var result = '';
    for (var i = 0; i < length; i++) {
        result += mask[Math.floor(Math.random() * Math.random() * 100000) % mask.length];
    }

    return result + Math.floor(Math.random() * 100000);
}

/**
 * @example
 * // Use pubsub for callback loop
 * pubsub.sub('loop', (i) => {
 *   console.log(i);
 *
 *   setImmediate(() => {
 *     pubsub.pub('loop', ++i);
 *   });
 * });
 *
 * pubsub.pub('loop', 0);
 *
 * // unsub event
 * let subToken = pubsub.sub('unsub', (data) => {
 *   console.log(data);
 * });
 *
 * pubsub.pub('unsub', 'hello world!');
 * pubsub.unSub('unsub', subToken);
 */
const PubSub = {
    _events: {},
    /**
     * Pub data to event handler
     * @param {string} event - event name
     * @param {var} data - data for handler
     */
    pub: function (event, data = '') {
        debug(`event ${event} receive ${JSON.stringify(data).substring(0, 200)} ...`);

        if (!this._events[event]) {
            return;
        }

        // https://nodejs.org/en/docs/guides/event-loop-timers-and-nexttick/
        setImmediate(() => {
            Object.keys(this._events[event]).map(token => this._events[event][token](data));
        });
    },
    /**
     * Sub event, register an event handler
     * @param {string} event - event name
     * @param {function} cb - handler
     * @return {string} token - sub token, which can be used for unsub
     */
    sub: function (event, cb) {
        if (!this._events[event]) {
            this._events[event] = {};
        }
        let token = createRandomStr(12);
        this._events[event][token] = cb;

        return token;
    },
    /**
     * unSub event
     * @param {string} event - event name
     * @param {string} token - the token, which you get from sub function
     */
    unSub: function (event, token) {
        if (this._events[event] && this._events[event][token]) {
            delete this._events[event][token];
        }
    }
};

module.exports = PubSub;